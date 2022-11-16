package com.thinksky.service;

import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.translate.TranslateException;
import com.thinksky.bert.BertZooModel;
import com.thinksky.classification.MakeModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.enterprise.context.ApplicationScoped;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

@ApplicationScoped
public class PredictService {

    private static final Logger logger = LoggerFactory.getLogger(PredictService.class);
    private final MakeModel makeModel;
    private final BertZooModel bertZooModel;

    public PredictService(MakeModel makeModel,
                          BertZooModel bertZooModel) {
        this.makeModel = makeModel;
        this.bertZooModel = bertZooModel;
    }

    public String getPrediction(String question) {

        try {
            var classificationGroup = makeModel.getClassificationGroup(question);

            logger.info("Classification Group: {}", classificationGroup);

            var resourceAsStream = Files.readAllBytes(Path.of("build/data/" + classificationGroup + ".txt"));
            var resourceDocument = new String(resourceAsStream);

            QAInput input = new QAInput(question, resourceDocument);

            return bertZooModel.predict(input).orElse("Not able predict");

        } catch (TranslateException | IOException e) {
            throw new RuntimeException(e);
        }
    }

}
