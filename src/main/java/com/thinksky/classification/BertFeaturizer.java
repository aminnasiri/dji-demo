package com.thinksky.classification;

import ai.djl.basicdataset.tabular.utils.DynamicBuffer;
import ai.djl.basicdataset.tabular.utils.Featurizer;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.bert.BertFullTokenizer;

import java.util.List;

public class BertFeaturizer implements Featurizer {

    private final BertFullTokenizer tokenizer;
    private final int maxLength; // the cut-off length

    public BertFeaturizer(BertFullTokenizer tokenizer, int maxLength) {
        this.tokenizer = tokenizer;
        this.maxLength = maxLength;
    }


    @Override
    public void featurize(DynamicBuffer buffer, String input) {

        Vocabulary vocab = tokenizer.getVocabulary();

        // convert sentence to tokens (toLowerCase for uncased model)
        List<String> tokens = tokenizer.tokenize(input.toLowerCase());

        // trim the tokens to maxLength
        tokens = tokens.size() > maxLength ? tokens.subList(0, maxLength) : tokens;

        // BERT embedding convention "[CLS] Your Sentence [SEP]"
        buffer.put(vocab.getIndex("[CLS]"));
        tokens.forEach(token -> buffer.put(vocab.getIndex(token)));
        buffer.put(vocab.getIndex("[SEP]"));
    }
}
