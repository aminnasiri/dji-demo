package com.thinksky.classification;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.tabular.CsvDataset;
import ai.djl.basicdataset.tabular.utils.Feature;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.bert.BertFullTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.PaddingStackBatchifier;
import ai.djl.translate.TranslateException;
import org.apache.commons.csv.CSVFormat;

import javax.enterprise.context.ApplicationScoped;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;

@ApplicationScoped
public class MyPrepareData {

    public MyPrepareData() {
    }

    public CsvDataset getDataset(int batchSize, BertFullTokenizer tokenizer, int maxLength, int limit) throws URISyntaxException {
        float paddingToken = tokenizer.getVocabulary().getIndex("[PAD]");

//        Path path = Path.of(this.getClass().getClassLoader().getResource("question_tag.csv").toURI());
        Path path = Paths.get("build/data/question_tag.csv");
        return CsvDataset.builder()
//                .optCsvUrl(amazonReview) // load from Url
                .optCsvFile(path)
                .setCsvFormat(CSVFormat.DEFAULT.withHeader()) // Setting TSV loading format
                .setSampling(batchSize, true) // make sample size and random access
                .optLimit(limit)
                .addFeature(
                        new Feature(
                                "question", new BertFeaturizer(tokenizer, maxLength)))
                .addLabel(
                        new Feature(
                                "tag", (buf, data) -> buf.put(Integer.parseInt(data) - 1)))
                .optDataBatchifier(
                        PaddingStackBatchifier.builder()
                                .optIncludeValidLengths(false)
                                .addPad(0, 0, (m) -> m.ones(new Shape(1)).mul(paddingToken))
                                .build()) // define how to pad dataset to a fix length
                .build();
    }


}
