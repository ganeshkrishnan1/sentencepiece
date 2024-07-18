#include "pretokenizer_for_training.h"

#include "testharness.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_join.h"
#include "third_party/absl/strings/str_split.h"
#include "trainer_interface.h"
#include "leveldb/db.h"

namespace sentencepiece {
namespace pretokenizer {

class MockPretokenizer : public PretokenizerForTrainingInterface {
 public:
  MockPretokenizer() : PretokenizerForTrainingInterface() {}
  ~MockPretokenizer() override = default;

  SentencePieceText Tokenize(absl::string_view text) const override {
    return spt_;
  }

  util::Status status() const override { return util::OkStatus(); }

  void SetOutput(const SentencePieceText &spt) { spt_ = spt; }

  // Expose db_ for testing
  leveldb::DB* GetDB() const { return db_.get(); }

 private:
  SentencePieceText spt_;
};

TEST(PretokenizerForTrainingTest, BaseTest) {
  MockPretokenizer mock;

  {
    SentencePieceText spt;
    spt.set_text("I love sentencepiece");
    auto *p1 = spt.add_pieces();
    p1->set_surface("I");
    p1->set_begin(0);
    p1->set_end(1);

    auto *p2 = spt.add_pieces();
    p2->set_surface("love");
    p2->set_begin(2);
    p2->set_end(6);

    auto *p3 = spt.add_pieces();
    p3->set_surface("sentence");
    p3->set_begin(7);
    p3->set_end(15);

    auto *p4 = spt.add_pieces();
    p4->set_surface("piece");
    p4->set_begin(15);
    p4->set_end(20);

    mock.SetOutput(spt);

    std::string key = mock.PreTokenize("I love sentencepiece");
    
    // Verify the content in LevelDB
    std::string value;
    leveldb::Status s = mock.GetDB()->Get(leveldb::ReadOptions(), "0", &value);
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(value, absl::StrCat("I", TrainerInterface::kWSStr, "love"));
    
    s = mock.GetDB()->Get(leveldb::ReadOptions(), "1", &value);
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(value, absl::StrCat("sentence", TrainerInterface::kWSStr, "piece"));
  }

  {
    SentencePieceText spt;
    spt.set_text("これはペンです");
    auto *p1 = spt.add_pieces();
    p1->set_surface("これ");
    p1->set_begin(0);
    p1->set_end(6);

    auto *p2 = spt.add_pieces();
    p2->set_surface("は");
    p2->set_begin(6);
    p2->set_end(9);

    auto *p3 = spt.add_pieces();
    p3->set_surface("ペン");
    p3->set_begin(9);
    p3->set_end(15);

    auto *p4 = spt.add_pieces();
    p4->set_surface("です");
    p4->set_begin(15);
    p4->set_end(21);

    mock.SetOutput(spt);

    std::string key = mock.PreTokenize("これはペンです");
    
    // Verify the content in LevelDB
    std::string value;
    leveldb::Status s = mock.GetDB()->Get(leveldb::ReadOptions(), "0", &value);
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(value, "これ");
    
    s = mock.GetDB()->Get(leveldb::ReadOptions(), "1", &value);
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(value, "は");
    
    s = mock.GetDB()->Get(leveldb::ReadOptions(), "2", &value);
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(value, "ペン");
    
    s = mock.GetDB()->Get(leveldb::ReadOptions(), "3", &value);
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(value, "です");
  }
}

}  // namespace pretokenizer
}  // namespace sentencepiece