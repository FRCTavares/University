/* Generated by the protocol buffer compiler.  DO NOT EDIT! */
/* Generated from: balls.proto */

#ifndef PROTOBUF_C_balls_2eproto__INCLUDED
#define PROTOBUF_C_balls_2eproto__INCLUDED

#include <protobuf-c/protobuf-c.h>

PROTOBUF_C__BEGIN_DECLS

#if PROTOBUF_C_VERSION_NUMBER < 1000000
# error This file was generated by a newer version of protoc-c which is incompatible with your libprotobuf-c headers. Please update your headers.
#elif 1004001 < PROTOBUF_C_MIN_COMPILER_VERSION
# error This file was generated by an older version of protoc-c which is incompatible with your libprotobuf-c headers. Please regenerate this file with a newer version of protoc-c.
#endif


typedef struct BallDrawDisplayMsg BallDrawDisplayMsg;


/* --- enums --- */


/* --- messages --- */

struct  BallDrawDisplayMsg
{
  ProtobufCMessage base;
  ProtobufCBinaryData ch;
  uint32_t x;
  uint32_t y;
};
#define BALL_DRAW_DISPLAY_MSG__INIT \
 { PROTOBUF_C_MESSAGE_INIT (&ball_draw_display_msg__descriptor) \
    , {0,NULL}, 0, 0 }


/* BallDrawDisplayMsg methods */
void   ball_draw_display_msg__init
                     (BallDrawDisplayMsg         *message);
size_t ball_draw_display_msg__get_packed_size
                     (const BallDrawDisplayMsg   *message);
size_t ball_draw_display_msg__pack
                     (const BallDrawDisplayMsg   *message,
                      uint8_t             *out);
size_t ball_draw_display_msg__pack_to_buffer
                     (const BallDrawDisplayMsg   *message,
                      ProtobufCBuffer     *buffer);
BallDrawDisplayMsg *
       ball_draw_display_msg__unpack
                     (ProtobufCAllocator  *allocator,
                      size_t               len,
                      const uint8_t       *data);
void   ball_draw_display_msg__free_unpacked
                     (BallDrawDisplayMsg *message,
                      ProtobufCAllocator *allocator);
/* --- per-message closures --- */

typedef void (*BallDrawDisplayMsg_Closure)
                 (const BallDrawDisplayMsg *message,
                  void *closure_data);

/* --- services --- */


/* --- descriptors --- */

extern const ProtobufCMessageDescriptor ball_draw_display_msg__descriptor;

PROTOBUF_C__END_DECLS


#endif  /* PROTOBUF_C_balls_2eproto__INCLUDED */
