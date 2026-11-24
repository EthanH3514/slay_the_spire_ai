#pragma once
// Encoder for the fixed, reviewed numeric public-state schema only. Not a
// general JSON serializer: keys are compile-time ASCII, values integers/bools.
#include <array>
#include <charconv>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace spire_session {
struct PublicScalar {
    std::int64_t value;
    bool boolean;
    template<class T> explicit PublicScalar(T v):value(v),boolean(std::is_same_v<T,bool>) {
        static_assert(std::is_integral_v<T>,"public numeric schema only");
    }
};
class PublicNumericJson {
    std::string text_;
public:
    PublicNumericJson(){text_.reserve(2048);}
    void raw(std::string_view text){text_.append(text.data(),text.size());}
    void key(std::string_view name){raw("\"");raw(name);raw("\":");}
    void scalar(PublicScalar v){
        if(v.boolean){raw(v.value?"true":"false");return;}
        char buffer[32];auto end=std::to_chars(buffer,buffer+sizeof(buffer),v.value);
        if(end.ec!=std::errc{})throw std::runtime_error("public_integer_encoding");
        text_.append(buffer,end.ptr);
    }
    template<class T> void value(T v){scalar(PublicScalar(v));}
    template<class T> void field(std::string_view name,T v){key(name);value(v);}
    std::string finish()&&{return std::move(text_);}
};
} // namespace spire_session
