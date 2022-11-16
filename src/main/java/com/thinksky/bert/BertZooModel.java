package com.thinksky.bert;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.enterprise.context.ApplicationScoped;
import java.io.IOException;
import java.util.Optional;

@ApplicationScoped
public class BertZooModel {

    private static final Logger log = LoggerFactory.getLogger(BertZooModel.class);

    public Optional<String> predict(QAInput input){

        Criteria<QAInput, String> criteria = Criteria.builder()
                .optApplication(Application.NLP.QUESTION_ANSWER)
                .setTypes(QAInput.class, String.class)
//                .optEngine("MXNet")
//                .optFilter("modelType", "distilbert")
                .optFilter("backbone", "bert")
                .optEngine("PyTorch") // Use PyTorch engine
                .optProgress(new ProgressBar()).build();
        try (ZooModel<QAInput, String> model = criteria.loadModel()) {
            Predictor<QAInput, String> predictor = model.newPredictor();
            String answer = predictor.predict(input);

            log.info("You are using: {} Engine", Engine.getInstance().getEngineName());
            log.info("Answer: {}", answer);

            return Optional.of(answer);
        } catch (ModelNotFoundException | MalformedModelException | IOException | TranslateException e) {
            throw new RuntimeException(e);
        }
    }
}
