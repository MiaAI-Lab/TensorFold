// The host half of the one-shot RoCE all-gather between 2 to 4 DGX Sparks (decode plan D1, work/research/R1-decode.md
// 3.1; TP=3/4 in work/zig-notes/r107-tpn.md).
//
// The protocol is b12x's "RoCEnante" (https://github.com/local-inference-lab/b12x, b12x/comm/roce, by the b12x
// contributors at Local Inference Lab, Apache License 2.0): each rank stages its shard into a pinned host send slot,
// rings a doorbell, and this proxy thread RDMA-writes the shard into every peer's receive slot over the RoCE rails,
// each rail's stripe followed on the same reliable queue pair by a 4-byte sequence number into the peer's flag for
// that stripe; the peer's kernel waits for every flag, then reads the shard. This file is a new implementation of that
// protocol for TensorFold's Zig CUDA runtime (zig/src/cuda/roce.zig drives it, zig/kernels/cuda/fn_roce.cu is the GPU
// half); the GLM-5.3 recipe's TensorFold patches 0006 (two ranks), 0066 (routes: one queue pair per peer and rail,
// a rail pairing by subnet), 0113 (the ring of four with relays) and 0114 (the ACK timeout) did the same for the Python
// engine (src/tensorfold/cuda/roce_proxy.c, by MiaAI-Lab); this file follows that design and its rules, in the Zig
// runtime's dlopen form.
//
// Region (pinned host memory mapped into the GPU, registered with every rail), offsets from tf_roce_layout:
//   recv[src][slot]            ROCE_SLOTS x world x slot_bytes   written by the peers' NICs
//   flag[src][slot][stripe]    128 B each, u32 sequence at 0, u32 payload length at 64 (ring only);
//                              `n_flag` stripes a peer (the same for every peer and rank)  written by the peers' NICs
//   send[slot]                 ROCE_SLOTS x slot_bytes             written by this rank's kernel
//   ctrl                       128 B: u32 seq (doorbell), u32 pad, u32 error seq, u32 missing peer,
//                              u32 bytes[ROCE_SLOTS], u32 missing stripe     written by this rank's kernel
//
// Routes: stripe s of the shard to peer p leaves on local rail route_local[p][s] for the peer's rail route_remote[p][s]
// (tf_roce_set_routes; the default is rail i to the peer's rail i, every rail: two ranks). Three ranks cabled to each
// other (a triangle, one subnet a cable) reach each peer over that cable's rails only.
//
// Ring (tf_roce_set_ring): four ranks cabled in rank order, each only to the previous and the next. A rank writes its
// shard to its two neighbours only, each stripe with a length word ahead of its flag, and relays the neighbours'
// stripes to the rank across (even stripes onward from the previous rank to the next, odd ones from the next to the
// previous), writing the flag of the original source, so every peer's flags arrive as without a ring and the GPU side
// is unchanged. A rank posts sequence s only after it has relayed, and the NIC has finished reading, everything up to
// s - 1: so a relayed slot is never overwritten while it is still being sent on.
//
// libibverbs is opened at run time (dlopen), so the binary runs without it until RoCE is turned on; only the
// queue-pair fast paths (post_send, poll_cq) go through the header's inline wrappers, which call the provider's
// function table and need no symbol.

#define _GNU_SOURCE
#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <ifaddrs.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>

#define ROCE_SLOTS 2
#define ROCE_HCAS_MAX 4
#define ROCE_WORLD_MAX 4
#define ROCE_FLAG_STRIDE 128
#define ROCE_LEN_OFFSET 64                 // a stripe flag's length word (ring), inside its ROCE_FLAG_STRIDE bytes
#define ROCE_IB_PORT 1
#define ROCE_DEPTH 256
#define ROCE_FWD_BIT ((uint64_t)1 << 32)   // wr_id of a relayed stripe
#define ROCE_ABI 7

// the exported libibverbs functions this file calls (by pointer: dlopen)
struct verbs {
  void *lib;
  struct ibv_device **(*get_device_list)(int *);
  void (*free_device_list)(struct ibv_device **);
  const char *(*get_device_name)(struct ibv_device *);
  struct ibv_context *(*open_device)(struct ibv_device *);
  int (*close_device)(struct ibv_context *);
  int (*query_port)(struct ibv_context *, uint8_t, struct ibv_port_attr *);
  int (*query_gid)(struct ibv_context *, uint8_t, int, union ibv_gid *);
  struct ibv_pd *(*alloc_pd)(struct ibv_context *);
  int (*dealloc_pd)(struct ibv_pd *);
  struct ibv_mr *(*reg_mr)(struct ibv_pd *, void *, size_t, int);
  int (*dereg_mr)(struct ibv_mr *);
  struct ibv_cq *(*create_cq)(struct ibv_context *, int, void *, struct ibv_comp_channel *, int);
  int (*destroy_cq)(struct ibv_cq *);
  struct ibv_qp *(*create_qp)(struct ibv_pd *, struct ibv_qp_init_attr *);
  int (*modify_qp)(struct ibv_qp *, struct ibv_qp_attr *, int);
  int (*destroy_qp)(struct ibv_qp *);
  const char *(*wc_status_str)(enum ibv_wc_status);
};

// what a rank tells the others to connect (gathered over the control link: every rank gets every rank's)
typedef struct {
  uint64_t region;  // the region's host address (the NIC's address space)
  uint32_t rkey[ROCE_HCAS_MAX];
  uint32_t mtu[ROCE_HCAS_MAX];
  uint32_t qpn[ROCE_HCAS_MAX][ROCE_WORLD_MAX];  // this rank's queue pair on each rail for each peer
  uint32_t n_hca;
  uint32_t slot_pages;  // slot_bytes / 4096: all ranks must use the same slots
  uint8_t gid[ROCE_HCAS_MAX][16];
} tf_roce_blob;

typedef struct {
  struct ibv_context *ctx;
  struct ibv_pd *pd;
  struct ibv_mr *mr;
  struct ibv_cq *cq;
  struct ibv_qp *qp[ROCE_WORLD_MAX];
  uint32_t outstanding[ROCE_WORLD_MAX];
  union ibv_gid gid;
  enum ibv_mtu mtu;
  uint64_t writes, bytes;
} hca_t;

typedef struct tf_roce {
  struct verbs v;
  int rank, world, n_hca, traffic_class;
  int gid_index[ROCE_HCAS_MAX];
  hca_t hca[ROCE_HCAS_MAX];
  // routes: stripe s to peer p leaves on local rail route_local[p][s] for the peer's rail route_remote[p][s]
  int n_stripe[ROCE_WORLD_MAX];
  int route_local[ROCE_WORLD_MAX][ROCE_HCAS_MAX];
  int route_remote[ROCE_WORLD_MAX][ROCE_HCAS_MAX];
  int n_flag;  // stripes (flags) a peer, the same for every peer and rank
  // ring: relayed[p] for the rank across; relay entry e sends stripe relay_stripe[e] of relay_src[e]'s payload on to
  // relay_dst[e], last done for sequence relay_last[e]
  int ring, relayed[ROCE_WORLD_MAX], n_relay;
  int relay_src[ROCE_HCAS_MAX], relay_stripe[ROCE_HCAS_MAX], relay_dst[ROCE_HCAS_MAX];
  uint32_t relay_last[ROCE_HCAS_MAX];
  uint64_t fwd_outstanding, fwd_posted;
  uint8_t *region;
  uint64_t region_bytes, slot_bytes;
  uint64_t recv_off, flag_off, send_off, ctrl_off;
  uint64_t peer_region[ROCE_WORLD_MAX];
  uint32_t peer_rkey[ROCE_HCAS_MAX][ROCE_WORLD_MAX];  // by stripe
  pthread_t thread;
  int thread_started, started;
  atomic_int running, failed;
  uint32_t posted_seq;
  _Atomic uint64_t ops;
  uint64_t writes;
  char err[512];
} tf_roce;

static void fail(tf_roce *r, const char *what, int e) {
  snprintf(r->err, sizeof(r->err), "%s: %s", what, e ? strerror(e) : "failed");
}

int tf_roce_abi(void) { return ROCE_ABI; }
uint64_t tf_roce_blob_bytes(void) { return sizeof(tf_roce_blob); }

// {recv, flag, send, ctrl, total} offsets for `world` ranks and slots of `slot_bytes` (a multiple of 4096)
int tf_roce_layout(int world, uint64_t slot_bytes, uint64_t *out) {
  if (world < 2 || world > ROCE_WORLD_MAX || slot_bytes == 0 || slot_bytes % 4096 != 0 || slot_bytes > (1ull << 30))
    return -1;
  const uint64_t recv = 0;
  const uint64_t flag = recv + (uint64_t)world * ROCE_SLOTS * slot_bytes;
  const uint64_t send = flag + (uint64_t)world * ROCE_SLOTS * ROCE_HCAS_MAX * ROCE_FLAG_STRIDE;
  const uint64_t ctrl = send + ROCE_SLOTS * slot_bytes;
  out[0] = recv;
  out[1] = flag;
  out[2] = send;
  out[3] = ctrl;
  out[4] = ctrl + ROCE_FLAG_STRIDE;
  return 0;
}

// The first RoCE v2 IPv4 GID of an RDMA device's port 1 (sysfs): its index, the address (host order) and the prefix
// length of the netdev that carries it (getifaddrs; 24 when not found). The Python engine's _ipv4_gid. 0 or -1.
int tf_roce_ipv4(const char *hca, int *gid_index, uint32_t *ip, int *prefix) {
  char path[512], text[128];
  int best = -1;
  uint32_t best_ip = 0;
  for (int i = 0; i < 256; i++) {
    snprintf(path, sizeof(path), "/sys/class/infiniband/%s/ports/1/gids/%d", hca, i);
    FILE *f = fopen(path, "r");
    if (!f) continue;
    char gid[64] = {0};
    const char *got = fgets(gid, sizeof(gid), f);
    fclose(f);
    if (!got) continue;
    snprintf(path, sizeof(path), "/sys/class/infiniband/%s/ports/1/gid_attrs/types/%d", hca, i);
    f = fopen(path, "r");
    if (!f) continue;
    text[0] = 0;
    if (!fgets(text, sizeof(text), f)) text[0] = 0;
    fclose(f);
    if (!strstr(text, "v2") && !strstr(text, "V2")) continue;
    // "0000:0000:0000:0000:0000:ffff:0a00:1601": an IPv4-mapped address
    unsigned w[8];
    if (sscanf(gid, "%x:%x:%x:%x:%x:%x:%x:%x", &w[0], &w[1], &w[2], &w[3], &w[4], &w[5], &w[6], &w[7]) != 8) continue;
    if (w[0] || w[1] || w[2] || w[3] || w[4] || w[5] != 0xffff) continue;
    const uint32_t a = (w[6] << 16) | w[7];
    if (a == 0) continue;
    if (best < 0) {
      best = i;
      best_ip = a;
    }
  }
  if (best < 0) return -1;
  *gid_index = best;
  *ip = best_ip;
  *prefix = 24;
  snprintf(path, sizeof(path), "/sys/class/infiniband/%s/ports/1/gid_attrs/ndevs/%d", hca, best);
  FILE *f = fopen(path, "r");
  if (f) {
    char dev[64] = {0};
    if (fgets(dev, sizeof(dev), f)) {
      dev[strcspn(dev, "\r\n ")] = 0;
      struct ifaddrs *list = NULL;
      if (getifaddrs(&list) == 0) {
        for (struct ifaddrs *p = list; p; p = p->ifa_next) {
          if (!p->ifa_addr || p->ifa_addr->sa_family != AF_INET || !p->ifa_netmask || strcmp(p->ifa_name, dev) != 0)
            continue;
          const uint32_t a = ntohl(((struct sockaddr_in *)p->ifa_addr)->sin_addr.s_addr);
          if (a != best_ip) continue;
          *prefix = __builtin_popcount(ntohl(((struct sockaddr_in *)p->ifa_netmask)->sin_addr.s_addr));
        }
        freeifaddrs(list);
      }
    }
    fclose(f);
  }
  return 0;
}

static int load_verbs(tf_roce *r) {
  struct verbs *v = &r->v;
  v->lib = dlopen("libibverbs.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!v->lib) {
    snprintf(r->err, sizeof(r->err), "dlopen libibverbs.so.1: %s", dlerror());
    return -1;
  }
#define SYM(field, name)                                                       \
  do {                                                                         \
    *(void **)&v->field = dlsym(v->lib, name);                                 \
    if (!v->field) {                                                           \
      snprintf(r->err, sizeof(r->err), "libibverbs.so.1 has no %s", name);     \
      return -1;                                                               \
    }                                                                          \
  } while (0)
  SYM(get_device_list, "ibv_get_device_list");
  SYM(free_device_list, "ibv_free_device_list");
  SYM(get_device_name, "ibv_get_device_name");
  SYM(open_device, "ibv_open_device");
  SYM(close_device, "ibv_close_device");
  SYM(query_port, "ibv_query_port");
  SYM(query_gid, "ibv_query_gid");
  SYM(alloc_pd, "ibv_alloc_pd");
  SYM(dealloc_pd, "ibv_dealloc_pd");
  SYM(reg_mr, "ibv_reg_mr");
  SYM(dereg_mr, "ibv_dereg_mr");
  SYM(create_cq, "ibv_create_cq");
  SYM(destroy_cq, "ibv_destroy_cq");
  SYM(create_qp, "ibv_create_qp");
  SYM(modify_qp, "ibv_modify_qp");
  SYM(destroy_qp, "ibv_destroy_qp");
  SYM(wc_status_str, "ibv_wc_status_str");
#undef SYM
  return 0;
}

// Rail h: the device, its protection domain and region registration, one completion queue, and one reliable queue
// pair for each peer (in INIT; tf_roce_connect moves the ones a route uses on).
static int open_hca(tf_roce *r, int h, const char *name) {
  struct verbs *v = &r->v;
  hca_t *p = &r->hca[h];
  int n = 0;
  struct ibv_device **list = v->get_device_list(&n);
  if (!list) {
    fail(r, "ibv_get_device_list", errno);
    return -1;
  }
  struct ibv_device *dev = NULL;
  for (int k = 0; k < n; k++)
    if (strcmp(v->get_device_name(list[k]), name) == 0) dev = list[k];
  if (dev) p->ctx = v->open_device(dev);
  v->free_device_list(list);
  if (!dev) {
    snprintf(r->err, sizeof(r->err), "no RDMA device %s", name);
    return -1;
  }
  if (!p->ctx) {
    fail(r, "ibv_open_device", errno);
    return -1;
  }
  struct ibv_port_attr pa;
  memset(&pa, 0, sizeof(pa));
  if (v->query_port(p->ctx, ROCE_IB_PORT, &pa) != 0) {
    fail(r, "ibv_query_port", errno);
    return -1;
  }
  if (pa.state != IBV_PORT_ACTIVE) {
    snprintf(r->err, sizeof(r->err), "%s port %d is not active", name, ROCE_IB_PORT);
    return -1;
  }
  p->mtu = pa.active_mtu;
  if (v->query_gid(p->ctx, ROCE_IB_PORT, r->gid_index[h], &p->gid) != 0) {
    fail(r, "ibv_query_gid", errno);
    return -1;
  }
  if (!(p->pd = v->alloc_pd(p->ctx))) {
    fail(r, "ibv_alloc_pd", errno);
    return -1;
  }
  if (!(p->mr = v->reg_mr(p->pd, r->region, r->region_bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE))) {
    fail(r, "ibv_reg_mr", errno);
    return -1;
  }
  if (!(p->cq = v->create_cq(p->ctx, ROCE_DEPTH * ROCE_WORLD_MAX, NULL, NULL, 0))) {
    fail(r, "ibv_create_cq", errno);
    return -1;
  }
  for (int peer = 0; peer < r->world; peer++) {
    if (peer == r->rank) continue;
    struct ibv_qp_init_attr qa;
    memset(&qa, 0, sizeof(qa));
    qa.send_cq = p->cq;
    qa.recv_cq = p->cq;
    qa.qp_type = IBV_QPT_RC;
    qa.cap.max_send_wr = ROCE_DEPTH;
    qa.cap.max_recv_wr = 1;
    qa.cap.max_send_sge = 1;
    qa.cap.max_recv_sge = 1;
    qa.cap.max_inline_data = 16;
    if (!(p->qp[peer] = v->create_qp(p->pd, &qa))) {
      fail(r, "ibv_create_qp", errno);
      return -1;
    }
    struct ibv_qp_attr a;
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_INIT;
    a.port_num = ROCE_IB_PORT;
    a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    const int rc = v->modify_qp(p->qp[peer], &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    if (rc != 0) {
      fail(r, "ibv_modify_qp INIT", rc);
      return -1;
    }
  }
  return 0;
}

void tf_roce_close(tf_roce *r);

// Opens `n_hca` RDMA devices (names[i]) on port 1 with GID `gid_index[i]`, registers `region`, makes one RC queue pair
// a rail and peer. Returns NULL with the reason in `err`. The default routes: rail i to the peer's rail i, every rail.
tf_roce *tf_roce_open(int world, int rank, const char *const *names, int n_hca, const int *gid_index,
                      int traffic_class, void *region, uint64_t region_bytes, uint64_t slot_bytes, char *err,
                      uint64_t err_len) {
  uint64_t lay[5];
  if (tf_roce_layout(world, slot_bytes, lay) != 0 || lay[4] > region_bytes || rank < 0 || rank >= world ||
      n_hca < 1 || n_hca > ROCE_HCAS_MAX || traffic_class < 0 || traffic_class > 255) {
    snprintf(err, err_len, "invalid RoCE geometry (world %d rank %d rails %d slot %llu region %llu)", world, rank,
             n_hca, (unsigned long long)slot_bytes, (unsigned long long)region_bytes);
    return NULL;
  }
  tf_roce *r = calloc(1, sizeof(*r));
  if (!r) {
    snprintf(err, err_len, "out of memory");
    return NULL;
  }
  r->rank = rank;
  r->world = world;
  r->n_hca = n_hca;
  r->traffic_class = traffic_class;
  for (int h = 0; h < n_hca; h++) r->gid_index[h] = gid_index[h];
  for (int p = 0; p < world; p++) {
    r->n_stripe[p] = p == rank ? 0 : n_hca;
    for (int h = 0; h < n_hca; h++) r->route_local[p][h] = r->route_remote[p][h] = h;
  }
  r->n_flag = n_hca;
  r->region = region;
  r->region_bytes = region_bytes;
  r->slot_bytes = slot_bytes;
  r->recv_off = lay[0];
  r->flag_off = lay[1];
  r->send_off = lay[2];
  r->ctrl_off = lay[3];
  if (load_verbs(r) != 0) goto bad;
  for (int h = 0; h < n_hca; h++)
    if (open_hca(r, h, names[h]) != 0) goto bad;
  return r;
bad:
  snprintf(err, err_len, "%s", r->err);
  tf_roce_close(r);
  return NULL;
}

// Before tf_roce_connect: peer's `n` stripes over (local[s], remote[s]) rails; `n_flag`: the stripes every peer has.
int tf_roce_set_routes(tf_roce *r, int peer, int n, const int *local, const int *remote, int n_flag) {
  if (r->started || peer < 0 || peer >= r->world || peer == r->rank || n < 1 || n > ROCE_HCAS_MAX || n != n_flag ||
      n_flag > ROCE_HCAS_MAX) {
    snprintf(r->err, sizeof(r->err), "invalid RoCE route to rank %d (%d stripes, %d a peer)", peer, n, n_flag);
    return -1;
  }
  for (int s = 0; s < n; s++) {
    if (local[s] < 0 || local[s] >= r->n_hca || remote[s] < 0 || remote[s] >= ROCE_HCAS_MAX) {
      snprintf(r->err, sizeof(r->err), "invalid RoCE route to rank %d: stripe %d", peer, s);
      return -1;
    }
    r->route_local[peer][s] = local[s];
    r->route_remote[peer][s] = remote[s];
  }
  r->n_stripe[peer] = n;
  r->n_flag = n_flag;
  return 0;
}

// After tf_roce_set_routes to both neighbours, before tf_roce_connect: four ranks in a ring in rank order, each cabled
// to rank - 1 and rank + 1 only.
int tf_roce_set_ring(tf_roce *r) {
  const int w = r->world, prev = (r->rank + w - 1) % w, next = (r->rank + 1) % w, across = (r->rank + 2) % w;
  if (r->started || w != 4 || r->n_stripe[prev] != r->n_flag || r->n_stripe[next] != r->n_flag ||
      r->n_flag > ROCE_HCAS_MAX) {
    snprintf(r->err, sizeof(r->err), "RoCE ring: four ranks, routes to both neighbours with %d stripes each",
             r->n_flag);
    return -1;
  }
  r->ring = 1;
  r->relayed[across] = 1;
  r->n_stripe[across] = 0;
  r->n_relay = r->n_flag;
  for (int s = 0; s < r->n_flag; s++) {
    r->relay_stripe[s] = s;
    r->relay_src[s] = (s % 2 == 0) ? prev : next;
    r->relay_dst[s] = (s % 2 == 0) ? next : prev;
  }
  return 0;
}

int tf_roce_blob_of(tf_roce *r, void *out) {
  tf_roce_blob b;
  memset(&b, 0, sizeof(b));
  b.region = (uint64_t)(uintptr_t)r->region;
  b.n_hca = (uint32_t)r->n_hca;
  b.slot_pages = (uint32_t)(r->slot_bytes / 4096);
  for (int h = 0; h < r->n_hca; h++) {
    b.rkey[h] = r->hca[h].mr->rkey;
    b.mtu[h] = (uint32_t)r->hca[h].mtu;
    memcpy(b.gid[h], r->hca[h].gid.raw, 16);
    for (int p = 0; p < r->world; p++) b.qpn[h][p] = p == r->rank ? 0 : r->hca[h].qp[p]->qp_num;
  }
  memcpy(out, &b, sizeof(b));
  return 0;
}

// The ACK timeout is 4.096 us x 2^t a try, 7 tries: 14 (b12x's) gave up after ~0.5 s without ACKs, which a four-Spark
// ring's link once stalled for while NCCL (its default 20: ~4 s a try) rode it out (recipe patch 0114).
static uint8_t ib_timeout(void) {
  const char *v = getenv("TF_ROCE_IB_TIMEOUT");
  int t = (v && *v) ? atoi(v) : 20;
  return (uint8_t)((t >= 1 && t <= 31) ? t : 20);
}

// local rail h's queue pair for peer p <-> the peer's rail rh's queue pair for this rank
static int connect_qp(tf_roce *r, int h, int p, const tf_roce_blob *b, int rh) {
  hca_t *hca = &r->hca[h];
  struct ibv_qp_attr a;
  memset(&a, 0, sizeof(a));
  a.qp_state = IBV_QPS_RTR;
  a.path_mtu = (enum ibv_mtu)(b->mtu[rh] < (uint32_t)hca->mtu ? b->mtu[rh] : (uint32_t)hca->mtu);
  a.dest_qp_num = b->qpn[rh][r->rank];
  a.rq_psn = 0;
  a.max_dest_rd_atomic = 1;
  a.min_rnr_timer = 12;
  a.ah_attr.is_global = 1;
  a.ah_attr.port_num = ROCE_IB_PORT;
  memcpy(a.ah_attr.grh.dgid.raw, b->gid[rh], 16);
  a.ah_attr.grh.sgid_index = (uint8_t)r->gid_index[h];
  a.ah_attr.grh.hop_limit = 64;
  a.ah_attr.grh.traffic_class = (uint8_t)r->traffic_class;
  int rc = r->v.modify_qp(hca->qp[p], &a,
                          IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                              IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
  if (rc != 0) {
    fail(r, "ibv_modify_qp RTR", rc);
    return -1;
  }
  memset(&a, 0, sizeof(a));
  a.qp_state = IBV_QPS_RTS;
  a.timeout = ib_timeout();
  a.retry_cnt = 7;
  a.rnr_retry = 7;
  a.sq_psn = 0;
  a.max_rd_atomic = 1;
  rc = r->v.modify_qp(hca->qp[p], &a,
                      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                          IBV_QP_MAX_QP_RD_ATOMIC);
  if (rc != 0) {
    fail(r, "ibv_modify_qp RTS", rc);
    return -1;
  }
  return 0;
}

// Every rank's blob, in rank order (this rank's own entry is read for nothing): each route's queue pair to RTR and RTS
// toward the peer's queue pair for this rank on the routed rail. `len` is world blobs.
int tf_roce_connect(tf_roce *r, const void *blobs, uint64_t len) {
  if (len != (uint64_t)r->world * sizeof(tf_roce_blob)) {
    snprintf(r->err, sizeof(r->err), "RoCE connect: %llu blob bytes for %d ranks", (unsigned long long)len, r->world);
    return -1;
  }
  const tf_roce_blob *all = (const tf_roce_blob *)blobs;
  for (int p = 0; p < r->world; p++) {
    if (p == r->rank) continue;
    if ((uint64_t)all[p].slot_pages * 4096 != r->slot_bytes) {
      snprintf(r->err, sizeof(r->err), "rank %d's RoCE slots are %u KiB, this rank's %llu KiB", p,
               all[p].slot_pages * 4, (unsigned long long)(r->slot_bytes >> 10));
      return -1;
    }
    r->peer_region[p] = all[p].region;
    if (r->relayed[p]) continue;
    if (r->n_stripe[p] != r->n_flag) {
      snprintf(r->err, sizeof(r->err), "rank %d has %d stripes, every peer needs %d", p, r->n_stripe[p], r->n_flag);
      return -1;
    }
    for (int s = 0; s < r->n_stripe[p]; s++) {
      const int h = r->route_local[p][s], rh = r->route_remote[p][s];
      if (rh >= (int)all[p].n_hca) {
        snprintf(r->err, sizeof(r->err), "rank %d has no rail %d (it opened %u)", p, rh, all[p].n_hca);
        return -1;
      }
      r->peer_rkey[s][p] = all[p].rkey[rh];
      if (connect_qp(r, h, p, &all[p], rh) != 0) return -1;
    }
  }
  return 0;
}

// Completions of signalled flag writes (one a stripe); any error fails the proxy.
static int reap(tf_roce *r, int h) {
  struct ibv_wc wc[32];
  const int n = ibv_poll_cq(r->hca[h].cq, 32, wc);
  if (n < 0) {
    fail(r, "ibv_poll_cq", errno);
    return -1;
  }
  for (int k = 0; k < n; k++) {
    if (wc[k].status != IBV_WC_SUCCESS) {
      snprintf(r->err, sizeof(r->err), "RDMA write to rank %u on rail %d failed: %s (vendor error 0x%x) after seq %u",
               (unsigned)(wc[k].wr_id & 0xffu), h, r->v.wc_status_str(wc[k].status), wc[k].vendor_err, r->posted_seq);
      return -1;
    }
    r->hca[h].outstanding[wc[k].wr_id & 0xffu]--;
    if (wc[k].wr_id & ROCE_FWD_BIT) r->fwd_outstanding--;
    r->hca[h].writes++;
    r->writes++;
  }
  return 0;
}

static uint64_t flag_index(const tf_roce *r, int src, uint32_t slot, int stripe) {
  return ((uint64_t)src * ROCE_SLOTS + slot) * (uint64_t)r->n_flag + (uint64_t)stripe;
}

// stripe s's [byte offset, bytes) of an n-byte payload split in 16-byte units over n_stripe stripes
static void stripe_range(uint32_t nbytes, int n_stripe, int s, uint64_t *offset, uint32_t *bytes) {
  const uint32_t total = nbytes / 16, base = total / (uint32_t)n_stripe, extra = total % (uint32_t)n_stripe;
  uint32_t start = 0;
  for (int i = 0; i < s; i++) start += base + ((uint32_t)i < extra ? 1u : 0u);
  *offset = (uint64_t)start * 16;
  *bytes = (base + ((uint32_t)s < extra ? 1u : 0u)) * 16;
}

// Sequence `seq`'s shard (`bytes`, a multiple of 16) from send[seq % SLOTS] into every directly reached peer's
// recv[rank][seq % SLOTS], split over the route's stripes in 16-byte units, each stripe then its flag (same queue
// pair, so in order); in a ring a length word goes ahead of each flag, for the neighbour that relays the stripe on.
static int post(tf_roce *r, uint32_t seq, uint32_t bytes) {
  if (bytes == 0 || bytes % 16 != 0 || bytes > r->slot_bytes) {
    snprintf(r->err, sizeof(r->err), "RoCE shard of %u bytes (a positive multiple of 16 up to %llu)", bytes,
             (unsigned long long)r->slot_bytes);
    return -1;
  }
  const uint32_t slot = seq % ROCE_SLOTS;
  uint8_t *send = r->region + r->send_off + (uint64_t)slot * r->slot_bytes;
  for (int p = 0; p < r->world; p++) {
    if (p == r->rank || r->relayed[p]) continue;
    for (int s = 0; s < r->n_stripe[p]; s++) {
      const int h = r->route_local[p][s];
      hca_t *hca = &r->hca[h];
      // the queue stays at most a quarter full so a provider that needs extra entries can never fail a post
      while (hca->outstanding[p] >= ROCE_DEPTH / 4) {
        if (reap(r, h) != 0) return -1;
        if (!atomic_load_explicit(&r->running, memory_order_relaxed)) {
          snprintf(r->err, sizeof(r->err), "stopped with %u writes outstanding to rank %d", hca->outstanding[p], p);
          return -1;
        }
      }
      uint64_t off;
      uint32_t stripe_bytes;
      stripe_range(bytes, r->n_stripe[p], s, &off, &stripe_bytes);
      uint32_t value = seq, len = bytes;
      const uint64_t remote = r->peer_region[p];
      struct ibv_sge fs = {.addr = (uint64_t)(uintptr_t)&value, .length = 4, .lkey = 0};
      struct ibv_send_wr fw;
      memset(&fw, 0, sizeof(fw));
      fw.wr_id = (uint64_t)p;
      fw.sg_list = &fs;
      fw.num_sge = 1;
      fw.opcode = IBV_WR_RDMA_WRITE;
      fw.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
      fw.wr.rdma.remote_addr = remote + r->flag_off + flag_index(r, r->rank, slot, s) * ROCE_FLAG_STRIDE;
      fw.wr.rdma.rkey = r->peer_rkey[s][p];
      struct ibv_send_wr *first = &fw;
      struct ibv_sge ls = {.addr = (uint64_t)(uintptr_t)&len, .length = 4, .lkey = 0};
      struct ibv_send_wr lw;
      if (r->ring) {
        memset(&lw, 0, sizeof(lw));
        lw.wr_id = (uint64_t)p;
        lw.next = &fw;
        lw.sg_list = &ls;
        lw.num_sge = 1;
        lw.opcode = IBV_WR_RDMA_WRITE;
        lw.send_flags = IBV_SEND_INLINE;
        lw.wr.rdma.remote_addr = fw.wr.rdma.remote_addr + ROCE_LEN_OFFSET;
        lw.wr.rdma.rkey = r->peer_rkey[s][p];
        first = &lw;
      }
      struct ibv_sge ds;
      struct ibv_send_wr dw;
      if (stripe_bytes != 0) {
        ds.addr = (uint64_t)(uintptr_t)(send + off);
        ds.length = stripe_bytes;
        ds.lkey = hca->mr->lkey;
        memset(&dw, 0, sizeof(dw));
        dw.wr_id = (uint64_t)p;
        dw.next = first;
        dw.sg_list = &ds;
        dw.num_sge = 1;
        dw.opcode = IBV_WR_RDMA_WRITE;
        dw.wr.rdma.remote_addr = remote + r->recv_off + ((uint64_t)r->rank * ROCE_SLOTS + slot) * r->slot_bytes + off;
        dw.wr.rdma.rkey = r->peer_rkey[s][p];
        first = &dw;
      }
      struct ibv_send_wr *bad = NULL;
      const int rc = ibv_post_send(hca->qp[p], first, &bad);
      if (rc != 0) {
        fail(r, "ibv_post_send", rc);
        return -1;
      }
      hca->outstanding[p]++;
      hca->bytes += stripe_bytes;
    }
  }
  atomic_fetch_add_explicit(&r->ops, 1, memory_order_relaxed);
  for (int h = 0; h < r->n_hca; h++)
    if (reap(r, h) != 0) return -1;
  return 0;
}

// Relay entry e's stripe of sequence seq (arrived here in recv[src][slot]) on to dst, at the same place there.
static int forward(tf_roce *r, int e, uint32_t seq) {
  const int src = r->relay_src[e], s = r->relay_stripe[e], dst = r->relay_dst[e];
  const uint32_t slot = seq % ROCE_SLOTS;
  const uint64_t fi = flag_index(r, src, slot, s);
  volatile uint32_t *lenp = (volatile uint32_t *)(r->region + r->flag_off + fi * ROCE_FLAG_STRIDE + ROCE_LEN_OFFSET);
  const uint32_t nbytes = __atomic_load_n(lenp, __ATOMIC_ACQUIRE);
  if (nbytes == 0 || nbytes % 16 != 0 || nbytes > r->slot_bytes) {
    snprintf(r->err, sizeof(r->err), "RoCE ring: rank %d's payload length %u (seq %u) is invalid", src, nbytes, seq);
    return -1;
  }
  uint64_t off;
  uint32_t bytes;
  stripe_range(nbytes, r->n_flag, s, &off, &bytes);
  const int h = r->route_local[dst][s];
  hca_t *hca = &r->hca[h];
  while (hca->outstanding[dst] >= ROCE_DEPTH / 4) {
    if (reap(r, h) != 0) return -1;
    if (!atomic_load_explicit(&r->running, memory_order_relaxed)) {
      snprintf(r->err, sizeof(r->err), "stopped while relaying to rank %d", dst);
      return -1;
    }
  }
  uint32_t value = seq;
  const uint64_t remote = r->peer_region[dst];
  struct ibv_sge fs = {.addr = (uint64_t)(uintptr_t)&value, .length = 4, .lkey = 0};
  struct ibv_send_wr fw;
  memset(&fw, 0, sizeof(fw));
  fw.wr_id = (uint64_t)dst | ROCE_FWD_BIT;
  fw.sg_list = &fs;
  fw.num_sge = 1;
  fw.opcode = IBV_WR_RDMA_WRITE;
  fw.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
  fw.wr.rdma.remote_addr = remote + r->flag_off + fi * ROCE_FLAG_STRIDE;
  fw.wr.rdma.rkey = r->peer_rkey[s][dst];
  struct ibv_send_wr dw, *first = &fw;
  struct ibv_sge ds;
  if (bytes != 0) {
    const uint64_t at = r->recv_off + ((uint64_t)src * ROCE_SLOTS + slot) * r->slot_bytes + off;
    ds.addr = (uint64_t)(uintptr_t)(r->region + at);
    ds.length = bytes;
    ds.lkey = hca->mr->lkey;
    memset(&dw, 0, sizeof(dw));
    dw.wr_id = (uint64_t)dst | ROCE_FWD_BIT;
    dw.next = &fw;
    dw.sg_list = &ds;
    dw.num_sge = 1;
    dw.opcode = IBV_WR_RDMA_WRITE;
    dw.wr.rdma.remote_addr = remote + at;
    dw.wr.rdma.rkey = r->peer_rkey[s][dst];
    first = &dw;
  }
  struct ibv_send_wr *bad = NULL;
  const int rc = ibv_post_send(hca->qp[dst], first, &bad);
  if (rc != 0) {
    fail(r, "ibv_post_send relay", rc);
    return -1;
  }
  hca->outstanding[dst]++;
  hca->bytes += bytes;
  r->fwd_outstanding++;
  r->fwd_posted++;
  return 0;
}

// Relay every stripe that has arrived (until 0), or every one up to sequence `until`, waiting for those (they have
// arrived: this rank's kernel for that sequence waited on them). Returns how many were relayed, -1 on failure.
static int relay(tf_roce *r, uint32_t until) {
  int done = 0;
  for (int e = 0; e < r->n_relay; e++) {
    for (;;) {
      const uint32_t seq = r->relay_last[e] + 1;
      if (until != 0 && (int32_t)(seq - until) > 0) break;
      volatile uint32_t *flag = (volatile uint32_t *)(r->region + r->flag_off +
                                                      flag_index(r, r->relay_src[e], seq % ROCE_SLOTS, r->relay_stripe[e]) *
                                                          ROCE_FLAG_STRIDE);
      if (__atomic_load_n(flag, __ATOMIC_ACQUIRE) != seq) {
        if (until == 0) break;
        if (!atomic_load_explicit(&r->running, memory_order_relaxed)) {
          snprintf(r->err, sizeof(r->err), "stopped waiting to relay rank %d's seq %u", r->relay_src[e], seq);
          return -1;
        }
        continue;
      }
      if (forward(r, e, seq) != 0) return -1;
      r->relay_last[e] = seq;
      done++;
    }
  }
  return done;
}

// Every relayed stripe read out by the NIC (its signalled flag completed).
static int relays_drained(tf_roce *r) {
  while (r->fwd_outstanding != 0) {
    for (int h = 0; h < r->n_hca; h++)
      if (reap(r, h) != 0) return -1;
    if (!atomic_load_explicit(&r->running, memory_order_relaxed)) {
      snprintf(r->err, sizeof(r->err), "stopped with %llu relays outstanding", (unsigned long long)r->fwd_outstanding);
      return -1;
    }
  }
  return 0;
}

static uint64_t now_ns(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

// The proxy: poll the doorbell; post every sequence it has not posted, in order. A kernel finishes on the peers'
// shards alone, so the next one can ring before this thread saw the last: at most ROCE_SLOTS are ever pending (a rank
// cannot get two gathers ahead of its peers), each with its slot's byte count. In a ring it also relays the neighbours'
// stripes as they arrive. Hot while gathers flow; after a second without one it naps 50 us between polls.
static void *proxy(void *arg) {
  tf_roce *r = arg;
  volatile uint32_t *ctrl = (volatile uint32_t *)(r->region + r->ctrl_off);
  uint64_t polls = 0, last = now_ns();
  int napping = 0;
  const struct timespec nap = {0, 50000};
  while (atomic_load_explicit(&r->running, memory_order_relaxed)) {
    if (r->ring) {
      const int n = relay(r, 0);
      if (n < 0) goto failed;
      if (n > 0) {
        last = now_ns();
        napping = 0;
      }
    }
    const uint32_t seq = __atomic_load_n(&ctrl[0], __ATOMIC_ACQUIRE);
    if (seq == r->posted_seq) {
      if ((++polls & 4095) == 0) {
        for (int h = 0; h < r->n_hca; h++)
          if (reap(r, h) != 0) goto failed;
        if (!napping && now_ns() - last > 1000000000ull) napping = 1;
      }
      if (napping) nanosleep(&nap, NULL);
      continue;
    }
    napping = 0;
    last = now_ns();
    const uint32_t pending = seq - r->posted_seq;
    if (pending > ROCE_SLOTS) {
      snprintf(r->err, sizeof(r->err), "the doorbell skipped %u sequences (posted %u, rung %u)", pending,
               r->posted_seq, seq);
      goto failed;
    }
    for (uint32_t s = r->posted_seq + 1; s != seq + 1; s++) {
      // ring: sequence s - 1's relays posted and read out first, so no neighbour's s + 1 (which waits on this
      // payload) can overwrite a slot that is still being relayed
      if (r->ring && s != 1 && (relay(r, s - 1) < 0 || relays_drained(r) != 0)) goto failed;
      if (post(r, s, ctrl[4 + s % ROCE_SLOTS]) != 0) goto failed;
      r->posted_seq = s;
    }
  }
  return NULL;
failed:
  atomic_store(&r->failed, 1);
  return NULL;
}

int tf_roce_start(tf_roce *r) {
  if (r->thread_started) return 0;
  volatile uint32_t *ctrl = (volatile uint32_t *)(r->region + r->ctrl_off);
  r->posted_seq = ctrl[0];
  for (int e = 0; e < r->n_relay; e++) r->relay_last[e] = r->posted_seq;
  r->started = 1;
  atomic_store(&r->failed, 0);
  atomic_store(&r->running, 1);
  const int rc = pthread_create(&r->thread, NULL, proxy, r);
  if (rc != 0) {
    atomic_store(&r->running, 0);
    fail(r, "pthread_create", rc);
    return -1;
  }
  r->thread_started = 1;
  return 0;
}

int tf_roce_failed(tf_roce *r) { return atomic_load(&r->failed); }
const char *tf_roce_error(tf_roce *r) { return r->err; }
uint64_t tf_roce_ops(tf_roce *r) { return atomic_load_explicit(&r->ops, memory_order_relaxed); }
uint64_t tf_roce_relayed(tf_roce *r) { return r->fwd_posted; }

void tf_roce_close(tf_roce *r) {
  if (!r) return;
  if (r->thread_started) {
    atomic_store(&r->running, 0);
    pthread_join(r->thread, NULL);
  }
  for (int h = 0; h < ROCE_HCAS_MAX; h++) {
    hca_t *p = &r->hca[h];
    for (int q = 0; q < ROCE_WORLD_MAX; q++)
      if (p->qp[q]) r->v.destroy_qp(p->qp[q]);
    if (p->cq) r->v.destroy_cq(p->cq);
    if (p->mr) r->v.dereg_mr(p->mr);
    if (p->pd) r->v.dealloc_pd(p->pd);
    if (p->ctx) r->v.close_device(p->ctx);
  }
  if (r->v.lib) dlclose(r->v.lib);
  free(r);
}
