/* Generated by the protocol buffer compiler.  DO NOT EDIT! */
/* Generated from: example.proto */

#ifndef PROTOBUF_C_example_2eproto__INCLUDED
#define PROTOBUF_C_example_2eproto__INCLUDED

#include <protobuf-c/protobuf-c.h>

PROTOBUF_C__BEGIN_DECLS

#if PROTOBUF_C_VERSION_NUMBER < 1000000
# error This file was generated by a newer version of protoc-c which is incompatible with your libprotobuf-c headers. Please update your headers.
#elif 1004001 < PROTOBUF_C_MIN_COMPILER_VERSION
# error This file was generated by an older version of protoc-c which is incompatible with your libprotobuf-c headers. Please regenerate this file with a newer version of protoc-c.
#endif


typedef struct SimpleMessage SimpleMessage;


/* --- enums --- */

typedef enum _EnumType {
  ENUM_TYPE__ENUM_ONE = 0,
  ENUM_TYPE__ENUM_TWO = 1
    PROTOBUF_C__FORCE_ENUM_TO_BE_INT_SIZE(ENUM_TYPE)
} EnumType;

/* --- messages --- */

struct  SimpleMessage
{
  ProtobufCMessage base;
  char *str_value;
  protobuf_c_boolean has_int_number;
  int32_t int_number;
  size_t n_float_array;
  float *float_array;
  protobuf_c_boolean has_enum_value;
  EnumType enum_value;
};
#define SIMPLE_MESSAGE__INIT \
 { PROTOBUF_C_MESSAGE_INIT (&simple_message__descriptor) \
    , NULL, 0, 10, 0,NULL, 0, ENUM_TYPE__ENUM_ONE }


/* SimpleMessage methods */
void   simple_message__init
                     (SimpleMessage         *message);
size_t simple_message__get_packed_size
                     (const SimpleMessage   *message);
size_t simple_message__pack
                     (const SimpleMessage   *message,
                      uint8_t             *out);
size_t simple_message__pack_to_buffer
                     (const SimpleMessage   *message,
                      ProtobufCBuffer     *buffer);
SimpleMessage *
       simple_message__unpack
                     (ProtobufCAllocator  *allocator,
                      size_t               len,
                      const uint8_t       *data);
void   simple_message__free_unpacked
                     (SimpleMessage *message,
                      ProtobufCAllocator *allocator);
/* --- per-message closures --- */

typedef void (*SimpleMessage_Closure)
                 (const SimpleMessage *message,
                  void *closure_data);

/* --- services --- */


/* --- descriptors --- */

extern const ProtobufCEnumDescriptor    enum_type__descriptor;
extern const ProtobufCMessageDescriptor simple_message__descriptor;

PROTOBUF_C__END_DECLS


#endif  /* PROTOBUF_C_example_2eproto__INCLUDED */
