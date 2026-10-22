//! Typed tool parameters: XML parameter text decoded by the offered schema (``tool_parameters.py``).
const std = @import("std");
const json = @import("json");
const reply_text = @import("reply_text.zig");
const tool_specs = @import("tool_specs.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// Each offered tool's parameter schemas by lowercase name.
pub const Schemas = std.StringHashMapUnmanaged(*const json.Object);

pub fn schemas(a: Allocator, tools: []const Value) Allocator.Error!Schemas {
    var out: Schemas = .empty;
    const empty = try json.newObject(a);
    for (tools) |tool| {
        if (tool != .object) continue;
        const function = if (tool.get("function")) |f| (if (f == .object) f else tool) else tool;
        var parameters: Value = .{ .object = empty };
        if (function.get("parameters")) |p| if (p.truthy()) {
            parameters = p;
        };
        if (!parameters.truthy()) if (function.get("input_schema")) |p| if (p.truthy()) {
            parameters = p;
        };
        var properties: Value = .{ .object = empty };
        if (parameters == .object) if (parameters.get("properties")) |p| if (p.truthy()) {
            properties = p;
        };
        const kept = try json.newObject(a);
        if (properties == .object) for (properties.object.keys(), properties.object.values()) |k, v| {
            if (v == .object) try kept.put(a, k, v);
        };
        const name = if (function.get("name")) |n| try tool_specs.pyStr(a, n) else "";
        try out.put(a, try std.ascii.allocLowerString(a, name), kept);
    }
    return out;
}

/// The schema of ``key`` in tool ``name``, or an empty one.
pub fn schemaOf(s: *const Schemas, name: []const u8, key: []const u8, a: Allocator) Allocator.Error!Value {
    const lower = try std.ascii.allocLowerString(a, name);
    const props = s.get(lower) orelse return .{ .object = try json.newObject(a) };
    return props.get(key) orelse .{ .object = try json.newObject(a) };
}

/// The JSON types a parameter schema admits, as bits (``schema_types`` of the recipe's tool_parameters.py).
pub const Types = u8;
const t_array: Types = 1;
const t_object: Types = 2;
const t_boolean: Types = 4;
const t_integer: Types = 8;
const t_number: Types = 16;
const t_null: Types = 32;
const t_string: Types = 64;
const t_other: Types = 128; // a type name this decoder does not know: never typed
const t_typed: Types = t_array | t_object | t_boolean | t_integer | t_number | t_null;

fn nameBit(name: []const u8) Types {
    const names = [_]struct { []const u8, Types }{ .{ "array", t_array }, .{ "object", t_object }, .{ "boolean", t_boolean }, .{ "integer", t_integer }, .{ "number", t_number }, .{ "null", t_null }, .{ "string", t_string } };
    for (names) |n| if (std.mem.eql(u8, n[0], name)) return n[1];
    return t_other;
}

/// ``_KINDS.get(type(v), "string")``: the kind of an enum or const value.
fn valueBit(v: Value) Types {
    return switch (v) {
        .bool => t_boolean,
        .int => t_integer,
        .float => t_number,
        .string => t_string,
        .null => t_null,
        .array => t_array,
        .object => t_object,
    };
}

/// ``schema_types``: its ``type`` (a name or a list of names), else those of its ``anyOf`` / ``oneOf`` members (none if
/// a member is untyped), else those of its ``enum`` values or ``const``.
pub fn schemaTypes(schema: Value) Types {
    if (schema != .object) return 0;
    if (schema.get("type")) |kind| switch (kind) {
        .string => |name| return nameBit(name),
        .array => |names| {
            var out: Types = 0;
            for (names) |n| if (n == .string) {
                out |= nameBit(n.string);
            };
            return out;
        },
        else => {},
    };
    for ([_][]const u8{ "anyOf", "oneOf" }) |key| {
        const members = schema.get(key) orelse continue;
        if (members != .array or members.array.len == 0) continue;
        var out: Types = 0;
        for (members.array) |m| {
            const found = schemaTypes(m);
            if (found == 0) return 0;
            out |= found;
        }
        return out;
    }
    if (schema.get("enum")) |values| if (values == .array and values.array.len > 0) {
        var out: Types = 0;
        for (values.array) |v| out |= valueBit(v);
        return out;
    };
    if (schema.get("const")) |c| return valueBit(c);
    return 0;
}

/// ``typed_parameter``: every type the schema admits is a JSON one other than string (a union with string, such as
/// ``["string", "null"]``, keeps the text as written).
pub fn typedParameter(schema: Value) bool {
    const kinds = schemaTypes(schema);
    return kinds != 0 and kinds & ~t_typed == 0;
}

/// ``nullable_text``: a text value that may also be null (``["string", "null"]``, pydantic's ``Optional[str]``).
pub fn nullableText(schema: Value) bool {
    const kinds = schemaTypes(schema);
    return kinds & t_string != 0 and kinds & t_null != 0;
}

fn admitted(kinds: Types, v: Value) bool {
    return (kinds & t_array != 0 and v == .array) or
        (kinds & t_object != 0 and v == .object) or
        (kinds & t_boolean != 0 and v == .bool) or
        (kinds & t_integer != 0 and v == .int) or
        (kinds & t_number != 0 and (v == .int or v == .float)) or
        (kinds & t_null != 0 and v == .null);
}

/// ``json.dumps(v, allow_nan=False)`` succeeds: no NaN or infinity anywhere.
pub fn finite(v: Value) bool {
    return switch (v) {
        .float => |f| std.math.isFinite(f),
        .array => |items| for (items) |i| {
            if (!finite(i)) break false;
        } else true,
        .object => |o| for (o.values()) |i| {
            if (!finite(i)) break false;
        } else true,
        else => true,
    };
}

/// ``closed_json``: the arrays and objects ``text`` left open, closed; null unless it only stops short of them.
pub fn closedJson(a: Allocator, text: []const u8) Allocator.Error!?[]const u8 {
    var closers: std.ArrayList(u8) = .empty;
    var in_string = false;
    var escaped = false;
    for (text) |ch| {
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else {
                escaped = ch == '\\';
                in_string = ch != '"';
            }
        } else if (ch == '"') {
            in_string = true;
        } else if (ch == '[' or ch == '{') {
            try closers.append(a, if (ch == '[') ']' else '}');
        } else if (ch == ']' or ch == '}') {
            if (closers.items.len == 0 or closers.pop().? != ch) return null;
        }
    }
    if (closers.items.len == 0 or in_string) return null;
    std.mem.reverse(u8, closers.items);
    return try std.mem.concat(a, u8, &.{ reply_text.pyRstrip(text), closers.items });
}

/// ``decode_parameter``: a string schema keeps the text (a nullable one reads ``null`` as null); a typed one reads JSON
/// of an admitted type, or the value closed (#87), and with ``python`` (Qwen's parsers) Python's spelling of one;
/// anything else stays as written, so a resent history renders the tokens the model wrote.
pub fn decode(a: Allocator, value: []const u8, schema: Value, python: bool) Allocator.Error!Value {
    if (!typedParameter(schema)) return if (nullableText(schema) and std.mem.eql(u8, value, "null")) .null else .{ .string = value };
    const kinds = schemaTypes(schema);
    const closed: ?[]const u8 = if (kinds & (t_array | t_object) != 0) try closedJson(a, value) else null;
    for ([_]?[]const u8{ value, closed }) |candidate| {
        const text = candidate orelse continue;
        const parsed = switch (try json.parseText(a, text)) {
            .ok => |v| v,
            .err => continue,
        };
        if (!finite(parsed)) continue;
        return if (admitted(kinds, parsed)) parsed else .{ .string = value };
    }
    if (!python) return .{ .string = value };
    const parsed = (try pythonLiteral(a, value)) orelse return .{ .string = value };
    if (!finite(parsed)) return .{ .string = value };
    return if (admitted(kinds, parsed)) parsed else .{ .string = value };
}

/// ``_python_literal``: True, None, quoted strings, numbers, lists, tuples and dicts; null if ``text`` is not one.
pub fn pythonLiteral(a: Allocator, text: []const u8) Allocator.Error!?Value {
    const stripped = reply_text.pyStrip(text);
    var lower_buf: [8]u8 = undefined;
    if (stripped.len <= 5) {
        const lower = std.ascii.lowerString(&lower_buf, stripped);
        if (std.mem.eql(u8, lower, "true")) return .{ .bool = true };
        if (std.mem.eql(u8, lower, "false")) return .{ .bool = false };
        if (std.mem.eql(u8, lower, "none") or std.mem.eql(u8, lower, "null")) return .null;
    }
    var p: Literal = .{ .a = a, .s = stripped };
    const v = p.expr(0) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Invalid => return null,
    };
    p.space();
    return if (p.i == p.s.len) v else null;
}

const Literal = struct {
    a: Allocator,
    s: []const u8,
    i: usize = 0,

    const Error = error{ Invalid, OutOfMemory };

    fn space(p: *Literal) void {
        while (p.i < p.s.len and (p.s[p.i] == ' ' or p.s[p.i] == '\t' or p.s[p.i] == '\n' or p.s[p.i] == '\r')) p.i += 1;
    }

    fn peek(p: *Literal) ?u8 {
        p.space();
        return if (p.i < p.s.len) p.s[p.i] else null;
    }

    fn expr(p: *Literal, depth: usize) Error!Value {
        if (depth > 200) return error.Invalid;
        const c = p.peek() orelse return error.Invalid;
        switch (c) {
            '[' => return p.sequence(']', depth),
            '(' => return p.sequence(')', depth),
            '{' => return p.dict(depth),
            '\'', '"' => return p.strings(),
            'r', 'R', 'u', 'U' => if (p.i + 1 < p.s.len and (p.s[p.i + 1] == '\'' or p.s[p.i + 1] == '"')) return p.strings(),
            else => {},
        }
        if (std.mem.startsWith(u8, p.s[p.i..], "None") and !p.wordAt(p.i + 4)) return p.word(4, .null);
        if (std.mem.startsWith(u8, p.s[p.i..], "True") and !p.wordAt(p.i + 4)) return p.word(4, .{ .bool = true });
        if (std.mem.startsWith(u8, p.s[p.i..], "False") and !p.wordAt(p.i + 5)) return p.word(5, .{ .bool = false });
        return p.number();
    }

    fn wordAt(p: *Literal, at: usize) bool {
        return at < p.s.len and (std.ascii.isAlphanumeric(p.s[at]) or p.s[at] == '_');
    }

    fn word(p: *Literal, n: usize, v: Value) Value {
        p.i += n;
        return v;
    }

    fn sequence(p: *Literal, close: u8, depth: usize) Error!Value {
        p.i += 1;
        var items: std.ArrayList(Value) = .empty;
        var commas: usize = 0;
        while (true) {
            if (p.peek() == close) {
                p.i += 1;
                if (close == ')' and items.items.len == 1 and commas == 0) return items.items[0]; // (x) is x
                return .{ .array = items.items };
            }
            try items.append(p.a, try p.expr(depth + 1));
            const next = p.peek() orelse return error.Invalid;
            if (next == ',') {
                p.i += 1;
                commas += 1;
            } else if (next != close) return error.Invalid;
        }
    }

    fn dict(p: *Literal, depth: usize) Error!Value {
        p.i += 1;
        const o = try json.newObject(p.a);
        if (p.peek() == '}') {
            p.i += 1;
            return .{ .object = o };
        }
        while (true) {
            const key = try p.expr(depth + 1);
            if (p.peek() != ':') return error.Invalid; // a set is no JSON value
            p.i += 1;
            const v = try p.expr(depth + 1);
            const name: []const u8 = switch (key) {
                .string => |s| s,
                .int => |t| t,
                .bool => |b| if (b) "true" else "false",
                .null => "null",
                .float => |f| blk: {
                    var buf: [40]u8 = undefined;
                    break :blk try p.a.dupe(u8, json.floatRepr(&buf, f));
                },
                else => return error.Invalid,
            };
            try o.put(p.a, name, v);
            const next = p.peek() orelse return error.Invalid;
            if (next == ',') {
                p.i += 1;
                if (p.peek() == '}') {
                    p.i += 1;
                    return .{ .object = o };
                }
            } else if (next == '}') {
                p.i += 1;
                return .{ .object = o };
            } else return error.Invalid;
        }
    }

    /// One or more adjacent string literals, joined as Python's parser joins them.
    fn strings(p: *Literal) Error!Value {
        var out: std.ArrayList(u8) = .empty;
        var any = false;
        while (p.peek()) |c| {
            var raw = false;
            if (c == 'r' or c == 'R' or c == 'u' or c == 'U') {
                if (p.i + 1 >= p.s.len or (p.s[p.i + 1] != '\'' and p.s[p.i + 1] != '"')) break;
                raw = c == 'r' or c == 'R';
                p.i += 1;
            } else if (c != '\'' and c != '"') break;
            try p.string(&out, raw);
            any = true;
        }
        if (!any) return error.Invalid;
        return .{ .string = out.items };
    }

    fn string(p: *Literal, out: *std.ArrayList(u8), raw: bool) Error!void {
        const q = p.s[p.i];
        const triple = p.i + 2 < p.s.len and p.s[p.i + 1] == q and p.s[p.i + 2] == q;
        p.i += if (triple) 3 else 1;
        while (true) {
            if (p.i >= p.s.len) return error.Invalid;
            const c = p.s[p.i];
            if (c == q and (!triple or (p.i + 2 < p.s.len and p.s[p.i + 1] == q and p.s[p.i + 2] == q))) {
                p.i += if (triple) 3 else 1;
                return;
            }
            if (c == '\n' and !triple) return error.Invalid;
            if (c != '\\') {
                try out.append(p.a, c);
                p.i += 1;
                continue;
            }
            if (p.i + 1 >= p.s.len) return error.Invalid;
            const e = p.s[p.i + 1];
            if (raw) {
                try out.appendSlice(p.a, p.s[p.i .. p.i + 2]);
                p.i += 2;
                continue;
            }
            p.i += 2;
            const simple: ?u8 = switch (e) {
                'n' => '\n',
                't' => '\t',
                'r' => '\r',
                '\\' => '\\',
                '\'' => '\'',
                '"' => '"',
                'a' => 7,
                'b' => 8,
                'f' => 12,
                'v' => 11,
                '0'...'7' => blk: {
                    var v: u21 = e - '0';
                    var n: usize = 1;
                    while (n < 3 and p.i < p.s.len and p.s[p.i] >= '0' and p.s[p.i] <= '7') : (n += 1) {
                        v = v * 8 + (p.s[p.i] - '0');
                        p.i += 1;
                    }
                    try appendCode(p.a, out, v);
                    break :blk null;
                },
                'x', 'u', 'U' => blk: {
                    const n: usize = if (e == 'x') 2 else if (e == 'u') 4 else 8;
                    if (p.i + n > p.s.len) return error.Invalid;
                    const v = std.fmt.parseInt(u21, p.s[p.i .. p.i + n], 16) catch return error.Invalid;
                    p.i += n;
                    try appendCode(p.a, out, v);
                    break :blk null;
                },
                '\n' => null,
                else => blk: {
                    try out.appendSlice(p.a, &.{ '\\', e });
                    break :blk null;
                },
            };
            if (simple) |ch| try out.append(p.a, ch);
        }
    }

    fn appendCode(a: Allocator, out: *std.ArrayList(u8), v: u21) Error!void {
        if (v > 0x10ffff) return error.Invalid;
        var buf: [4]u8 = undefined;
        const n = json.wtf8Encode(v, &buf);
        try out.appendSlice(a, buf[0..n]);
    }

    fn number(p: *Literal) Error!Value {
        var negative = false;
        while (p.peek()) |c| {
            if (c == '-') negative = !negative else if (c != '+') break;
            p.i += 1;
        }
        const start = p.i;
        while (p.i < p.s.len and (std.ascii.isAlphanumeric(p.s[p.i]) or p.s[p.i] == '_' or p.s[p.i] == '.' or
            ((p.s[p.i] == '+' or p.s[p.i] == '-') and p.i > start and (p.s[p.i - 1] == 'e' or p.s[p.i - 1] == 'E')))) p.i += 1;
        const text = p.s[start..p.i];
        if (text.len == 0) return error.Invalid;
        const sign: []const u8 = if (negative) "-" else "";
        if (text.len > 2 and text[0] == '0' and std.ascii.findIgnoreCase("xob", text[1..2]) != null) {
            const base: u8 = switch (std.ascii.toLower(text[1])) {
                'x' => 16,
                'o' => 8,
                else => 2,
            };
            const digits = std.mem.replaceOwned(u8, p.a, text[2..], "_", "") catch return error.OutOfMemory;
            const v = std.fmt.parseInt(i128, digits, base) catch return error.Invalid;
            return .{ .int = try std.fmt.allocPrint(p.a, "{s}{d}", .{ sign, v }) };
        }
        if (std.mem.indexOfAny(u8, text, ".eE") == null) {
            const digits = std.mem.replaceOwned(u8, p.a, text, "_", "") catch return error.OutOfMemory;
            for (digits) |ch| if (!std.ascii.isDigit(ch)) return error.Invalid;
            if (digits.len > 1 and digits[0] == '0' and std.mem.trimStart(u8, digits, "0").len > 0) return error.Invalid;
            const trimmed = std.mem.trimStart(u8, digits, "0");
            if (trimmed.len == 0) return .{ .int = "0" };
            return .{ .int = try std.mem.concat(p.a, u8, &.{ sign, trimmed }) };
        }
        if (std.mem.indexOfAny(u8, text, "jJ") != null) return error.Invalid; // complex
        const f = @import("fields.zig").pyFloat(text) orelse return error.Invalid;
        if (std.mem.indexOfAny(u8, text, "infaINFA") != null and !std.ascii.isDigit(text[0]) and text[0] != '.') return error.Invalid;
        return .{ .float = if (negative) -f else f };
    }
};

test "decode parameters" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const int_schema = (try json.parse(a, "{\"type\": \"integer\"}")).ok;
    try std.testing.expectEqualStrings("5", (try decode(a, "5", int_schema, true)).int);
    try std.testing.expectEqualStrings("5.5", (try decode(a, "5.5", int_schema, true)).string);
    const arr = (try json.parse(a, "{\"type\": \"array\"}")).ok;
    const closed = try decode(a, "[1, [2]", arr, true);
    try std.testing.expectEqual(@as(usize, 2), closed.array.len);
    const py = try decode(a, "['a', None, True]", arr, true);
    try std.testing.expectEqualStrings("[\"a\", null, true]", try json.stringify(a, py, .{}));
    try std.testing.expectEqual(Value.null, (try decode(a, " None ", (try json.parse(a, "{\"type\": \"null\"}")).ok, true)));
}

fn schemaOfText(a: Allocator, text: []const u8) !Value {
    return (try json.parse(a, text)).ok;
}

test "schemas typed by a type list, anyOf, enum or const decode as the recipe's tool_parameters.py does" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // ["integer", "null"]: typed, a number or null
    const opt = try schemaOfText(a, "{\"type\":[\"integer\",\"null\"]}");
    try std.testing.expect(typedParameter(opt));
    try std.testing.expectEqualStrings("7", (try decode(a, "7", opt, false)).int);
    try std.testing.expect((try decode(a, "null", opt, false)) == .null);
    // ["string", "null"]: the text as written, except a bare null
    const nullable = try schemaOfText(a, "{\"type\":[\"string\",\"null\"]}");
    try std.testing.expect(!typedParameter(nullable) and nullableText(nullable));
    try std.testing.expect((try decode(a, "null", nullable, false)) == .null);
    try std.testing.expectEqualStrings("nullx", (try decode(a, "nullx", nullable, false)).string);
    try std.testing.expectEqualStrings("5", (try decode(a, "5", nullable, false)).string);
    // a plain string schema keeps "null" as text
    const plain = try schemaOfText(a, "{\"type\":\"string\"}");
    try std.testing.expectEqualStrings("null", (try decode(a, "null", plain, false)).string);
    // anyOf of typed members is typed; one untyped member makes none
    const any = try schemaOfText(a, "{\"anyOf\":[{\"type\":\"array\"},{\"type\":\"null\"}]}");
    try std.testing.expect(typedParameter(any));
    try std.testing.expectEqual(@as(usize, 2), (try decode(a, "[1,2]", any, false)).array.len);
    try std.testing.expect(!typedParameter(try schemaOfText(a, "{\"anyOf\":[{\"type\":\"array\"},{}]}")));
    // enum and const are typed by their values
    const en = try schemaOfText(a, "{\"enum\":[1,2,3]}");
    try std.testing.expect(typedParameter(en));
    try std.testing.expectEqualStrings("2", (try decode(a, "2", en, false)).int);
    const named = try schemaOfText(a, "{\"enum\":[\"a\",\"b\"]}");
    try std.testing.expect(!typedParameter(named));
    const fixed = try schemaOfText(a, "{\"const\":true}");
    try std.testing.expect((try decode(a, "true", fixed, false)).bool);
    // a type the decoder does not know is untyped
    try std.testing.expect(!typedParameter(try schemaOfText(a, "{\"type\":\"str\"}")));
    // a typed value of another type stays as written
    const arr = try schemaOfText(a, "{\"type\":\"array\"}");
    try std.testing.expectEqualStrings("{\"a\":1}", (try decode(a, "{\"a\":1}", arr, false)).string);
    // an array left one closer short is closed (#87)
    try std.testing.expectEqual(@as(usize, 2), (try decode(a, "[1,2", arr, false)).array.len);
}
