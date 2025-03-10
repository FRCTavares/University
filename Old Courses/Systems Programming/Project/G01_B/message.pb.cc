// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: message.proto

#include "message.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>

PROTOBUF_PRAGMA_INIT_SEG

namespace _pb = ::PROTOBUF_NAMESPACE_ID;
namespace _pbi = _pb::internal;

PROTOBUF_CONSTEXPR ScoreUpdate::ScoreUpdate(
    ::_pbi::ConstantInitialized): _impl_{
    /*decltype(_impl_.scores_)*/{}
  , /*decltype(_impl_._scores_cached_byte_size_)*/{0}
  , /*decltype(_impl_.characters_)*/{}
  , /*decltype(_impl_._cached_size_)*/{}} {}
struct ScoreUpdateDefaultTypeInternal {
  PROTOBUF_CONSTEXPR ScoreUpdateDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~ScoreUpdateDefaultTypeInternal() {}
  union {
    ScoreUpdate _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 ScoreUpdateDefaultTypeInternal _ScoreUpdate_default_instance_;
static ::_pb::Metadata file_level_metadata_message_2eproto[1];
static constexpr ::_pb::EnumDescriptor const** file_level_enum_descriptors_message_2eproto = nullptr;
static constexpr ::_pb::ServiceDescriptor const** file_level_service_descriptors_message_2eproto = nullptr;

const uint32_t TableStruct_message_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::ScoreUpdate, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::ScoreUpdate, _impl_.scores_),
  PROTOBUF_FIELD_OFFSET(::ScoreUpdate, _impl_.characters_),
};
static const ::_pbi::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, -1, sizeof(::ScoreUpdate)},
};

static const ::_pb::Message* const file_default_instances[] = {
  &::_ScoreUpdate_default_instance_._instance,
};

const char descriptor_table_protodef_message_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\rmessage.proto\"1\n\013ScoreUpdate\022\016\n\006scores"
  "\030\001 \003(\005\022\022\n\ncharacters\030\002 \003(\tb\006proto3"
  ;
static ::_pbi::once_flag descriptor_table_message_2eproto_once;
const ::_pbi::DescriptorTable descriptor_table_message_2eproto = {
    false, false, 74, descriptor_table_protodef_message_2eproto,
    "message.proto",
    &descriptor_table_message_2eproto_once, nullptr, 0, 1,
    schemas, file_default_instances, TableStruct_message_2eproto::offsets,
    file_level_metadata_message_2eproto, file_level_enum_descriptors_message_2eproto,
    file_level_service_descriptors_message_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::_pbi::DescriptorTable* descriptor_table_message_2eproto_getter() {
  return &descriptor_table_message_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY2 static ::_pbi::AddDescriptorsRunner dynamic_init_dummy_message_2eproto(&descriptor_table_message_2eproto);

// ===================================================================

class ScoreUpdate::_Internal {
 public:
};

ScoreUpdate::ScoreUpdate(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor(arena, is_message_owned);
  // @@protoc_insertion_point(arena_constructor:ScoreUpdate)
}
ScoreUpdate::ScoreUpdate(const ScoreUpdate& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  ScoreUpdate* const _this = this; (void)_this;
  new (&_impl_) Impl_{
      decltype(_impl_.scores_){from._impl_.scores_}
    , /*decltype(_impl_._scores_cached_byte_size_)*/{0}
    , decltype(_impl_.characters_){from._impl_.characters_}
    , /*decltype(_impl_._cached_size_)*/{}};

  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:ScoreUpdate)
}

inline void ScoreUpdate::SharedCtor(
    ::_pb::Arena* arena, bool is_message_owned) {
  (void)arena;
  (void)is_message_owned;
  new (&_impl_) Impl_{
      decltype(_impl_.scores_){arena}
    , /*decltype(_impl_._scores_cached_byte_size_)*/{0}
    , decltype(_impl_.characters_){arena}
    , /*decltype(_impl_._cached_size_)*/{}
  };
}

ScoreUpdate::~ScoreUpdate() {
  // @@protoc_insertion_point(destructor:ScoreUpdate)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void ScoreUpdate::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  _impl_.scores_.~RepeatedField();
  _impl_.characters_.~RepeatedPtrField();
}

void ScoreUpdate::SetCachedSize(int size) const {
  _impl_._cached_size_.Set(size);
}

void ScoreUpdate::Clear() {
// @@protoc_insertion_point(message_clear_start:ScoreUpdate)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  _impl_.scores_.Clear();
  _impl_.characters_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* ScoreUpdate::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // repeated int32 scores = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedInt32Parser(_internal_mutable_scores(), ptr, ctx);
          CHK_(ptr);
        } else if (static_cast<uint8_t>(tag) == 8) {
          _internal_add_scores(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr));
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // repeated string characters = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 18)) {
          ptr -= 1;
          do {
            ptr += 1;
            auto str = _internal_add_characters();
            ptr = ::_pbi::InlineGreedyStringParser(str, ptr, ctx);
            CHK_(ptr);
            CHK_(::_pbi::VerifyUTF8(str, "ScoreUpdate.characters"));
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<18>(ptr));
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

uint8_t* ScoreUpdate::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:ScoreUpdate)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated int32 scores = 1;
  {
    int byte_size = _impl_._scores_cached_byte_size_.load(std::memory_order_relaxed);
    if (byte_size > 0) {
      target = stream->WriteInt32Packed(
          1, _internal_scores(), byte_size, target);
    }
  }

  // repeated string characters = 2;
  for (int i = 0, n = this->_internal_characters_size(); i < n; i++) {
    const auto& s = this->_internal_characters(i);
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      s.data(), static_cast<int>(s.length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "ScoreUpdate.characters");
    target = stream->WriteString(2, s, target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:ScoreUpdate)
  return target;
}

size_t ScoreUpdate::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:ScoreUpdate)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated int32 scores = 1;
  {
    size_t data_size = ::_pbi::WireFormatLite::
      Int32Size(this->_impl_.scores_);
    if (data_size > 0) {
      total_size += 1 +
        ::_pbi::WireFormatLite::Int32Size(static_cast<int32_t>(data_size));
    }
    int cached_size = ::_pbi::ToCachedSize(data_size);
    _impl_._scores_cached_byte_size_.store(cached_size,
                                    std::memory_order_relaxed);
    total_size += data_size;
  }

  // repeated string characters = 2;
  total_size += 1 *
      ::PROTOBUF_NAMESPACE_ID::internal::FromIntSize(_impl_.characters_.size());
  for (int i = 0, n = _impl_.characters_.size(); i < n; i++) {
    total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
      _impl_.characters_.Get(i));
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_impl_._cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData ScoreUpdate::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSourceCheck,
    ScoreUpdate::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*ScoreUpdate::GetClassData() const { return &_class_data_; }


void ScoreUpdate::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg) {
  auto* const _this = static_cast<ScoreUpdate*>(&to_msg);
  auto& from = static_cast<const ScoreUpdate&>(from_msg);
  // @@protoc_insertion_point(class_specific_merge_from_start:ScoreUpdate)
  GOOGLE_DCHECK_NE(&from, _this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  _this->_impl_.scores_.MergeFrom(from._impl_.scores_);
  _this->_impl_.characters_.MergeFrom(from._impl_.characters_);
  _this->_internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void ScoreUpdate::CopyFrom(const ScoreUpdate& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:ScoreUpdate)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool ScoreUpdate::IsInitialized() const {
  return true;
}

void ScoreUpdate::InternalSwap(ScoreUpdate* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  _impl_.scores_.InternalSwap(&other->_impl_.scores_);
  _impl_.characters_.InternalSwap(&other->_impl_.characters_);
}

::PROTOBUF_NAMESPACE_ID::Metadata ScoreUpdate::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_message_2eproto_getter, &descriptor_table_message_2eproto_once,
      file_level_metadata_message_2eproto[0]);
}

// @@protoc_insertion_point(namespace_scope)
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::ScoreUpdate*
Arena::CreateMaybeMessage< ::ScoreUpdate >(Arena* arena) {
  return Arena::CreateMessageInternal< ::ScoreUpdate >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
