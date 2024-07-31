// Copyright 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.!

#include <cmath>
#include <string>

#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/strings/string_view.h"
#include "util.h"
#include "word_model.h"
#include "word_model_trainer.h"

#include <leveldb/db.h>
#include <leveldb/iterator.h>
#include "absl/strings/numbers.h"
#include <unordered_map>

#include "leveldb_utils.h"

namespace sentencepiece {
namespace word {

util::Status Trainer::Train() {
  CHECK(g_leveldb_manager.GetDB() != nullptr) << "LevelDB is not initialized";
  RETURN_IF_ERROR(status());

  CHECK_OR_RETURN(normalizer_spec_.escape_whitespaces());
  CHECK_EQ_OR_RETURN(TrainerSpec::WORD, trainer_spec_.model_type());

  RETURN_IF_ERROR(LoadSentences());

  std::unordered_map<std::string, int64_t> freq;

  std::unique_ptr<leveldb::Iterator> it(g_leveldb_manager.GetDB()->NewIterator(leveldb::ReadOptions()));
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    const std::string &sentence = it->key().ToString();
    const std::string &value = it->value().ToString();
    int64_t count;
    CHECK(absl::SimpleAtoi(value, &count)) << "Invalid count: " << value;
    for (const auto &s : SplitIntoWords(sentence)) {
      freq[std::string(s)] += count;
    }
  }
  
  const int vocab_size = trainer_spec_.vocab_size() - meta_pieces_.size();
  CHECK_GE_OR_RETURN(vocab_size, 0);

  uint64 sum = 0;
  for (const auto &it : freq) {
    sum += it.second;
  }

  const auto logsum = std::log(static_cast<float>(sum));

  CHECK_OR_RETURN(final_pieces_.empty());
  for (const auto &it : Sorted(freq)) {
    if (it.first.find(kUNKStr) != std::string::npos) {
      continue;
    }
    if (!trainer_spec_.use_all_vocab() &&
        final_pieces_.size() == static_cast<size_t>(vocab_size)) {
      break;
    }
    final_pieces_.emplace_back(
        it.first, std::log(static_cast<float>(it.second)) - logsum);
  }

  if (trainer_spec_.use_all_vocab()) {
    trainer_spec_.set_vocab_size(final_pieces_.size() + meta_pieces_.size());
  }

  return Save();
}
}  // namespace word
}  // namespace sentencepiece
