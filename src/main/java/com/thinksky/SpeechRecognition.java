package com.thinksky;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.audio.Audio;
import ai.djl.modality.audio.AudioFactory;
import ai.djl.modality.audio.translator.SpeechRecognitionTranslatorFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;

import javax.sound.sampled.UnsupportedAudioFileException;

/**
 * An example of inference using a speech recognition model.
 *
 * <p>See this <a
 * href="https://github.com/deepjavalibrary/djl/blob/master/examples/docs/speech_recognition.md">doc</a>
 * for information about this example.
 */
public final class SpeechRecognition {

    public static final Logger logger = LoggerFactory.getLogger((SpeechRecognition.class));

    private SpeechRecognition() {}

    public static void main(String[] args)
            throws UnsupportedAudioFileException, IOException, TranslateException, ModelException {
        System.out.println("Result: " + predict());
    }

    public static String predict()
            throws UnsupportedAudioFileException, IOException, ModelException, TranslateException {
        // Load model.
        // Wav2Vec2 model is a speech model that accepts a float array corresponding to the raw
        // waveform of the speech signal.
        String url = "https://resources.djl.ai/test-models/pytorch/wav2vec2.zip";
        Criteria<Audio, String> criteria =
                Criteria.builder()
                        .setTypes(Audio.class, String.class)
                        .optModelUrls(url)
                        .optTranslatorFactory(new SpeechRecognitionTranslatorFactory())
                        .optModelName("wav2vec2.ptl")
                        .optEngine("PyTorch")
                        .build();

        // Read in audio file
//        String wave = "https://resources.djl.ai/audios/speech.wav";
        var wave = "/Users/amin/Developments/appsrc/opensources/speech-recognition/vosk/vosk-server/client-samples/java/f1.wav";
        Audio audio = AudioFactory.getInstance().fromFile(Path.of(wave));
        try (ZooModel<Audio, String> model = criteria.loadModel();
             Predictor<Audio, String> predictor = model.newPredictor()) {
            return predictor.predict(audio);
        }
    }
}
