// JSON schema to GBNF through llama.cpp's own converter (common/json-schema-to-grammar.cpp at the
// pin), so `json_schema` requests give the grammar llama-server builds (ADR 0083 group B).
// build.rs compiles this file with the converter's sources from vendor/llama.cpp/common; libcommon
// itself is not built (LLAMA_BUILD_COMMON=OFF).
//
// The three string helpers below are copied from common/common.cpp at the pin
// (Copyright (c) 2023-2026 The ggml authors, MIT), because that file needs all of libcommon.

#include "json-schema-to-grammar.h"
#include "common.h"

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

std::string string_join(const std::vector<std::string> & values, const std::string & separator) {
    std::ostringstream result;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            result << separator;
        }
        result << values[i];
    }
    return result.str();
}

std::vector<std::string> string_split(const std::string & str, const std::string & delimiter) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = str.find(delimiter);

    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }

    parts.push_back(str.substr(start));

    return parts;
}

std::string string_repeat(const std::string & str, size_t n) {
    if (n == 0) {
        return "";
    }

    std::string result;
    result.reserve(str.length() * n);

    for (size_t i = 0; i < n; ++i) {
        result += str;
    }

    return result;
}

static char * dup_string(const std::string & s) {
    char * p = static_cast<char *>(std::malloc(s.size() + 1));
    if (p) {
        std::memcpy(p, s.c_str(), s.size() + 1);
    }
    return p;
}

// Returns 0 and the grammar in *out, or 1 and a message in *out. Free *out with tetraeder_schema_free.
extern "C" int tetraeder_json_schema_to_grammar(const char * schema_json, char ** out) {
    try {
        *out = dup_string(json_schema_to_grammar(common_json::parse(schema_json)));
        return 0;
    } catch (const std::exception & e) {
        *out = dup_string(e.what());
        return 1;
    } catch (...) {
        *out = dup_string("unknown error");
        return 1;
    }
}

extern "C" void tetraeder_schema_free(char * p) {
    std::free(p);
}
