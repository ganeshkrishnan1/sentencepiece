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

#include "bpe_model_trainer.h"

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

#include "pretokenizer_for_training.h"
#include "third_party/absl/container/flat_hash_set.h"
#include "third_party/absl/strings/str_join.h"
#include "third_party/absl/strings/str_replace.h"
#include "util.h"

#include <leveldb/db.h>
#include "leveldb_utils.h"

namespace sentencepiece {
namespace bpe {

// Constructor implementation
Trainer::Trainer(const TrainerSpec &trainer_spec,
                 const NormalizerSpec &normalizer_spec,
                 const NormalizerSpec &denormalizer_spec)
    : TrainerInterface(trainer_spec, normalizer_spec, denormalizer_spec) {
  // Any additional initialization can go here
}

// Destructor implementation
Trainer::~Trainer() {
}

// Implementation of OpenSentenceDB method
util::Status Trainer::OpenSentenceDB() {
  return g_leveldb_manager.OpenDB("sentence_db");
}

// Implementation of CloseSentenceDB method, if needed
util::Status Trainer::CloseSentenceDB() {
  return g_leveldb_manager.CloseDB();
}

std::string Trainer::Symbol::ToString() const {
  return string_util::UnicodeTextToUTF8(chars);
}

Trainer::Symbol *Trainer::GetCharSymbol(char32 c) {
  const uint64 freq = port::FindWithDefault(required_chars_, c, 1);
  CHECK_GT(freq, 0);
  const auto it = symbols_cache_.find(c);
  if (it != symbols_cache_.end()) {
    return it->second;
  }
  Symbol *s = new Symbol;
  allocated_.push_back(s);
  s->is_unk = (kUNKChar == c);
  s->fp = c;
  s->chars.push_back(c);
  s->freq = freq;
  port::InsertOrDie(&symbols_cache_, s->fp, s);
  return s;
}

Trainer::Symbol *Trainer::GetPairSymbol(const Symbol *left,
                                        const Symbol *right) {
  if (left == nullptr || right == nullptr || left->is_unk || right->is_unk) {
    return nullptr;
  }

  const uint64 fp = port::FingerprintCat(left->fp, right->fp);
  const auto it = symbols_cache_.find(fp);
  if (it != symbols_cache_.end()) {
    return it->second;
  }

  CHECK(!left->chars.empty());
  CHECK(!right->chars.empty());
  string_util::UnicodeText ut;
  for (const char32 c : left->chars) ut.push_back(c);
  for (const char32 c : right->chars) ut.push_back(c);

  // Do not make an invalid piece.
  if (!IsValidSentencePiece(ut)) {
    return nullptr;
  }

  Symbol *s = new Symbol;
  allocated_.push_back(s);
  s->fp = fp;
  s->left = left;
  s->right = right;
  s->chars = ut;
  port::InsertOrDie(&symbols_cache_, s->fp, s);
  return s;
}

void Trainer::ComputeFreq(Symbol *symbol) const {
  if (symbol->freq > 0) {  // if freq == 0, re-computation is required.
    return;
  }
  CHECK_EQ(0, symbol->freq);
  for (auto it = symbol->positions.begin(); it != symbol->positions.end();) {
    const Position pos = DecodePos(*it);
    // symbols_[sid][left] and symbols_[sid][right] must store
    // the same symbols in symbol->left and symbol->right.
    if (symbol->left != symbols_[pos.sid][pos.left] ||
        symbol->right != symbols_[pos.sid][pos.right]) {
      it = symbol->positions.erase(it);
    } else {
      std::string value;
      std::string key = std::to_string(pos.sid);
      leveldb::Status status = g_leveldb_manager.GetDB()->Get(leveldb::ReadOptions(), key, &value);
      if (status.ok()) {
        size_t tab_pos = value.find('\t');
        if (tab_pos != std::string::npos) {
          symbol->freq += std::stoll(value.substr(tab_pos + 1));
        }
      }
      ++it;
    }
  }
}

int Trainer::GetNextIndex(int sid, int index) const {
  for (size_t i = index + 1; i < symbols_[sid].size(); ++i) {
    if (symbols_[sid][i] == nullptr) continue;
    return i;
  }
  return -1;
}

int Trainer::GetPrevIndex(int sid, int index) const {
  for (int i = index - 1; i >= 0; --i) {
    if (symbols_[sid][i] == nullptr) continue;
    return i;
  }
  return -1;
}

void Trainer::AddNewPair(int sid, int left, int right) {
  if (left == -1 || right == -1) return;
  auto *symbol = GetPairSymbol(symbols_[sid][left], symbols_[sid][right]);
  if (symbol != nullptr) {
    active_symbols_.insert(symbol);
    symbol->positions.insert(EncodePos(sid, left, right));
  }
}

void Trainer::ResetFreq(int sid, int left, int right, const Symbol *best) {
  if (left == -1 || right == -1) return;
  auto *symbol = GetPairSymbol(symbols_[sid][left], symbols_[sid][right]);
  if (symbol != nullptr && symbol != best) {
    symbol->freq = 0;
  }
}

void Trainer::UpdateActiveSymbols() {
  std::vector<Symbol *> symbols;
  for (auto &it : symbols_cache_) {
    Symbol *symbol = it.second;
    if (symbol->IsBigram()) {
      ComputeFreq(symbol);
      symbols.push_back(symbol);
    }
  }

  // At least kMinActiveSymbolsSize symbols must be in |active_symbols_|.
  constexpr int kMinActiveSymbolsSize = 1000;

  // Keeps top 5% frequent symbols.
  constexpr float kTopFrequentRatio = 0.05;
  const int size =
      std::min<int>(std::max<int>(kMinActiveSymbolsSize,
                                  symbols_cache_.size() * kTopFrequentRatio),
                    symbols.size());

  std::partial_sort(symbols.begin(), symbols.begin() + size, symbols.end(),
                    [](Symbol *s1, Symbol *s2) { return s1->freq > s2->freq; });
  LOG(INFO) << "Updating active symbols. max_freq=" << symbols[0]->freq
            << " min_freq=" << symbols[size - 1]->freq;

  active_symbols_.clear();
  active_symbols_.insert(symbols.begin(), symbols.begin() + size);
}

util::Status Trainer::Train() {
  RETURN_IF_ERROR(OpenSentenceDB());
  RETURN_IF_ERROR(status());

  CHECK_OR_RETURN(normalizer_spec_.escape_whitespaces());
  CHECK_EQ_OR_RETURN(TrainerSpec::BPE, trainer_spec_.model_type());

  symbols_.clear();
  allocated_.clear();
  symbols_cache_.clear();
  active_symbols_.clear();

  // Load all sentences
  RETURN_IF_ERROR(LoadSentences());

  if (trainer_spec_.split_by_whitespace()) {
    SplitSentencesByWhitespace();
  }

  // Pretokenizer applied only in training time.
  // Pretokenizer is used as a constraint of piece extractions.
  const auto *pretokenizer = SentencePieceTrainer::GetPretokenizerForTraining();

  std::unique_ptr<leveldb::Iterator> it(g_leveldb_manager.GetDB()->NewIterator(leveldb::ReadOptions()));
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    std::string value = it->value().ToString();
    size_t tab_pos = value.find('\t');
    if (tab_pos != std::string::npos) {
      std::string sentence = value.substr(0, tab_pos);
      int64_t freq = std::stoll(value.substr(tab_pos + 1));
      
      // Apply pretokenizer if needed
      if (pretokenizer || !trainer_spec_.pretokenization_delimiter().empty()) {
        if (pretokenizer) {
          sentence = absl::StrJoin(pretokenizer->PreTokenize(sentence),
                                  TrainerInterface::kUPPBoundaryStr);
        } else if (!trainer_spec_.pretokenization_delimiter().empty()) {
          sentence = absl::StrReplaceAll(
              sentence, {{trainer_spec_.pretokenization_delimiter(), TrainerInterface::kUPPBoundaryStr}});
        }
      }

      // Initialize symbols for this sentence
      std::vector<Symbol*> sentence_symbols;
      for (const char32 c : string_util::UTF8ToUnicodeText(sentence)) {
        sentence_symbols.push_back(GetCharSymbol(c));
      }
      symbols_.push_back(sentence_symbols);

      // Make bigram symbols for this sentence
      for (size_t i = 1; i < sentence_symbols.size(); ++i) {
        AddNewPair(symbols_.size() - 1, i - 1, i);
      }
    }
  }

  const int vocab_size =
      trainer_spec_.vocab_size() - meta_pieces_.size() - required_chars_.size();
  CHECK_GE_OR_RETURN(vocab_size, 0);

  // We may see duplicated pieces that are extracted with different path.
  // In real segmentation phase, we can consider them as one symbol.
  // e.g., "aaa" => "aa" + "a" or "a" + "aa".
  absl::flat_hash_set<std::string> dup;

  // Main loop.
  CHECK_OR_RETURN(final_pieces_.empty());
  while (final_pieces_.size() < static_cast<size_t>(vocab_size)) {
    constexpr int kUpdateActiveSymbolsInteval = 100;
    if (final_pieces_.size() % kUpdateActiveSymbolsInteval == 0) {
      UpdateActiveSymbols();
    }

    // Scanning active symbols, finds the best_symbol with highest freq.
    Symbol *best_symbol = nullptr;
    for (auto &it : active_symbols_) {
      Symbol *symbol = it;
      ComputeFreq(symbol);
      // If the frequency is the same, take shorter symbol.
      // if the length is the same, use lexicographical comparison
      if (best_symbol == nullptr ||
          (symbol->freq > best_symbol->freq ||
           (symbol->freq == best_symbol->freq &&
            (symbol->chars.size() < best_symbol->chars.size() ||
             (symbol->chars.size() == best_symbol->chars.size() &&
              symbol->ToString() < best_symbol->ToString()))))) {
        best_symbol = symbol;
      }
    }

    if (best_symbol == nullptr) {
      LOG(WARNING) << "No valid symbol found";
      break;
    }

    if (!dup.insert(best_symbol->ToString()).second) {
      // Removes best_symbol so it is not selected again.
      symbols_cache_.erase(best_symbol->fp);
      active_symbols_.erase(best_symbol);
      continue;
    }

    // Stores the best_symbol in the final output.
    final_pieces_.emplace_back(best_symbol->ToString(),
                               -static_cast<float>(final_pieces_.size()));

    if (final_pieces_.size() % 20 == 0) {
      LOG(INFO) << "Added: freq=" << best_symbol->freq
                << " size=" << final_pieces_.size()
                << " all=" << symbols_cache_.size()
                << " active=" << active_symbols_.size()
                << " piece=" << best_symbol->ToString();
    }

    // Add new bigrams which are created after symbol replacement.
    // We do not need to scan all characters, but scan the neighbors in
    // best_symbol.
    for (const uint64 &encoded_pos : best_symbol->positions) {
      const Position pos = DecodePos(encoded_pos);

      if (symbols_[pos.sid][pos.left] == nullptr) {
        // left index might be NULL (set in the previous iteration)
        // when left_symbol == right_symbol.
        continue;
      }
      CHECK_OR_RETURN(symbols_[pos.sid][pos.right]);

      // We have three bigrams [prev, left], [left, right], [right, next],
      // which are affected with this symbol replacement.
      const int next = GetNextIndex(pos.sid, pos.right);
      const int prev = GetPrevIndex(pos.sid, pos.left);

      // Resets the frequencies of bigrams [prev, left] and [right, next].
      ResetFreq(pos.sid, prev, pos.left, best_symbol);
      ResetFreq(pos.sid, pos.right, next, best_symbol);

      // Merges two symbols.
      symbols_[pos.sid][pos.left] = best_symbol;
      symbols_[pos.sid][pos.right] = nullptr;

      // Makes new symbol bigrams [prev, left] and [left, next].
      AddNewPair(pos.sid, prev, pos.left);
      AddNewPair(pos.sid, pos.left, next);
    }

    // Removes best_symbol so it is not selected again.
    symbols_cache_.erase(best_symbol->fp);
    active_symbols_.erase(best_symbol);
  }  // end of main loop

  // Adds required_chars_
  for (const auto &w : Sorted(required_chars_)) {
    const Symbol *symbol = GetCharSymbol(w.first);
    final_pieces_.emplace_back(symbol->ToString(),
                               -static_cast<float>(final_pieces_.size()));
  }

  port::STLDeleteElements(&allocated_);

  return Save();
}
}  // namespace bpe
}  // namespace sentencepiece
