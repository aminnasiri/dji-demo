package com.thinksky.classification;

import ai.djl.modality.Classifications;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.bert.BertFullTokenizer;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.util.Arrays;
import java.util.List;

class MyTranslator implements Translator<String, Classifications> {

    private BertFullTokenizer tokenizer;
    private Vocabulary vocab;
    private List<String> ranks;

    public MyTranslator(BertFullTokenizer tokenizer) {
        this.tokenizer = tokenizer;
        vocab = tokenizer.getVocabulary();
        ranks = Arrays.asList("1", "2", "3", "4", "5");
    }

    @Override
    public Batchifier getBatchifier() {
        return Batchifier.STACK;
    }

    @Override
    public NDList processInput(TranslatorContext ctx, String input) {
        List<String> tokens = tokenizer.tokenize(input);
        float[] indices = new float[tokens.size() + 2];
        indices[0] = vocab.getIndex("[CLS]");
        for (int i = 0; i < tokens.size(); i++) {
            indices[i + 1] = vocab.getIndex(tokens.get(i));
        }
        indices[indices.length - 1] = vocab.getIndex("[SEP]");
        return new NDList(ctx.getNDManager().create(indices));
    }

    @Override
    public Classifications processOutput(TranslatorContext ctx, NDList list) {
        return new Classifications(ranks, list.singletonOrThrow().softmax(0));
    }
}

