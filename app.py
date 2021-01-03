from typing import Any

import ginza
import pandas as pd
import spacy
import streamlit as st
import streamlit.components.v1 as stc

DICT_POS_JP = {
    "ADJ": "形容詞", "ADP": "設置詞", "ADV": "副詞", "AUX": "助動詞", "CCONJ": "接続詞", "DET": "限定詞", "INTJ": "間投詞", "NOUN": "名詞",
    "NUM": "数詞", "PART": "助詞", "PRON": "代名詞", "PROPN": "固有名詞", "PUNCT": "句読点", "SCONJ": "連結詞", "SYM": "シンボル",
    "VERB": "動詞", "X": "その他"
}

DICT_DEP_JP = {"nsubj": "名詞句主語", "obj": "目的語", "iobj": "間接目的語", "csubj": "節主語", "ccomp": "節補語", "xcomp": "開いた節補語",
               "obl": "斜格要素", "vocative": "呼格要素", "expl": "虚辞", "dislocated": "転位された要素", "advcl": "副詞的修飾節",
               "advmod": "副詞修飾語", "discourse": "談話要素", "aux": "助動詞", "cop": "コピュラ", "mark": "節標識", "nmod": "名詞修飾語",
               "appos": "同格要素", "nummod": "数詞", "acl": "形容詞的修飾節", "amod": "形容詞修飾語", "det": "限定詞", "clf": "助数詞",
               "case": "格標識", "conj": "等位並列句・節", "cc": "等位接続詞", "fixed": "複合機能表現", "flat": "構造のない複単語表現",
               "compound": "複合語", "list": "リスト表現", "parataxis": "並置表現", "orphan": "主辞がない語", "goeswith": "単語区切り誤り",
               "reparandum": "言い直し", "punct": "句読点", "root": "文の主辞", "dep": "不明な依存関係"}


class Toc:
    def __init__(self):
        self._items = []
        self._placeholder = None

    def title(self, text):
        self._markdown(text, "h1")

    def header(self, text):
        self._markdown(text, "h2", " " * 2)

    def subheader(self, text):
        self._markdown(text, "h3", " " * 4)

    def placeholder(self, sidebar=False):
        self._placeholder = st.sidebar.empty() if sidebar else st.empty()

    def generate(self):
        if self._placeholder:
            self._placeholder.markdown("\n".join(self._items), unsafe_allow_html=True)

    def _markdown(self, text, level, space=""):
        key = "".join(filter(str.isalnum, text)).lower()

        st.markdown(f"<{level} id='{key}'>{text}</{level}>", unsafe_allow_html=True)
        self._items.append(f"{space}* <a href='#{key}'>{text}</a>")


def create_manual(sent: spacy.tokens.Span) -> Any:
    words = []
    arcs = []

    start_index = sent[0].i if sent else 0
    for token in sent:
        # original "tag": token.pos_
        words.append({"text": token.orth_, "tag": token.tag_})
        # original "label": token.dep_
        label = DICT_DEP_JP.get(token.dep_, token.dep_)
        if token.i == token.head.i:
            pass
        elif token.i < token.head.i:
            arcs.append(
                {"start": token.i - start_index, "end": token.head.i - start_index, "label": label, "dir": "left"})
        else:
            arcs.append(
                {"start": token.head.i - start_index, "end": token.i - start_index, "label": label, "dir": "right"})
    return {"words": words, "arcs": arcs}


def main():
    st.set_page_config(layout="wide", initial_sidebar_state="expanded")
    st.title("GiNZA NLP Library")
    toc = Toc()
    toc.placeholder(True)

    input_list = st.text_area("入力文字列",
                              '銀座でランチをご一緒しましょう。今度の日曜日はどうですか。\n吾輩は猫である。 名前はまだ無い。 ').splitlines()
    ignore_lf = st.checkbox("改行を無視して1回で解析する。", False)
    if not st.button("実行"):
        st.stop()
        return
    if ignore_lf:
        input_list = ["".join(input_list)]
    with st.spinner(f'Wait for it...'):
        nlp = spacy.load('ja_ginza')
        # time.sleep(1.0)
        for i, input_str in enumerate(input_list):
            doc = nlp(input_str)
            for j, sent in enumerate(doc.sents):
                toc.subheader(f"{i + 1}-{j + 1}. {sent}")
                svg2 = spacy.displacy.render(create_manual(sent), style="dep",
                                             options={"compact": True, "offset_x": 200, "distance": 175}, manual=True)
                st.image(svg2, width=(len(sent) + 1) * 120)
                df = pd.DataFrame(index=[],
                                  columns=["i(index)", "orth(テキスト)", "lemma(基本形)", "reading_form(読みカナ)",
                                           "pos(PartOfSpeech)", "pos(品詞)", "tag(品詞詳細)", "inflection(活用情報)",
                                           "ent_type(エンティティ型)",
                                           "ent_iob(エンティティIOB)", "lang(言語)", "dep(dependency)", "dep(構文従属関係)",
                                           "head.i(親index)", "bunsetu_bi_label",
                                           "bunsetu_position_type", "is_bunsetu_head", "ent_label_ontonotes",
                                           "ent_label_ene"])
                for token in sent:
                    row = pd.DataFrame([token.i, token.orth_, token.lemma_, ginza.reading_form(token), token.pos_,
                                        DICT_POS_JP.get(token.pos_, token.pos_), token.tag_,
                                        ginza.inflection(token) or "-", token.ent_type_ or "-", token.ent_iob_ or "-",
                                        token.lang_, token.dep_,
                                        DICT_DEP_JP.get(token.dep_, token.dep_),
                                        token.head.i,
                                        ginza.bunsetu_bi_label(token), ginza.bunsetu_position_type(token),
                                        ginza.is_bunsetu_head(token),
                                        ginza.ent_label_ontonotes(token) or "-", ginza.ent_label_ene(token) or "-",
                                        ], index=df.columns).T
                    df = df.append(row, ignore_index=True)
                st.table(df.T)
                st.subheader("文節区切り")
                bunsetu_list = ginza.bunsetu_spans(sent)
                st.text("/".join([bunsetu.orth_ for bunsetu in bunsetu_list]))
                st.subheader("文節の主辞区間と句の区分")
                st.text("/".join([f"{phrase}({phrase.label_})" for phrase in ginza.bunsetu_phrase_spans(sent)]))
                st.subheader("固有表現(エンティティ)")
                if sent.ents:
                    svg_ent = spacy.displacy.render(sent, style="ent")
                    stc.html(svg_ent)
                else:
                    st.text("No Entity")
                toc.generate()
    toc.generate()
    # st.balloons()


if __name__ == '__main__':
    main()
