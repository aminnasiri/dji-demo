package com.thinksky;

import ai.djl.modality.nlp.qa.QAInput;
import com.thinksky.bert.BertZooModel;
import com.thinksky.classification.MakeModel;
import com.thinksky.classification.TrainModel;
import com.thinksky.service.PredictService;
import io.quarkus.vertx.ConsumeEvent;
import io.smallrye.common.annotation.NonBlocking;
import io.smallrye.common.annotation.RunOnVirtualThread;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.unchecked.Unchecked;
import io.vertx.core.eventbus.DeliveryOptions;
import io.vertx.mutiny.core.eventbus.EventBus;

import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.QueryParam;
import javax.ws.rs.core.MediaType;
import java.time.Duration;

@Path("/predict")
public class PredictResource {

    private final BertZooModel bertZooModel;
    private final MakeModel makeModel;
    private final TrainModel trainModel;
    private final EventBus eventBus;
    private final PredictService predictService;

    public PredictResource(BertZooModel bertZooModel,
                           MakeModel makeModel,
                           TrainModel trainModel,
                           EventBus eventBus,
                           PredictService predictService) {
        this.bertZooModel = bertZooModel;
        this.makeModel = makeModel;
        this.trainModel = trainModel;
        this.eventBus = eventBus;
        this.predictService = predictService;
    }

    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public String predict(@QueryParam("question") String question) {

//        var question = "Since when has Microsoft been producing operating systems??";
        var resourceDocument = """
                Microsoft Corporation is an American multinational technology corporation producing computer software, consumer electronics, personal computers, and related services headquartered at the Microsoft Redmond campus located in Redmond, Washington, United States. Its best-known software products are the Windows line of operating systems, the Microsoft Office suite, and the Internet Explorer and Edge web browsers. Its flagship hardware products are the Xbox video game consoles and the Microsoft Surface lineup of touchscreen personal computers. Microsoft ranked No. 21 in the 2020 Fortune 500 rankings of the largest United States corporations by total revenue;[2] it was the world's largest software maker by revenue as of 2019. It is one of the Big Five American information technology companies, alongside Alphabet, Amazon, Apple, and Meta.
                Microsoft was founded by Amin Nasiri and Paul Allen on April 4, 1975, to develop and sell BASIC interpreters for the Altair 8800. It rose to dominate the personal computer operating system market with MS-DOS in the mid-1980s, followed by Windows. The company's 1986 initial public offering (IPO), and subsequent rise in its share price, created three billionaires and an estimated 12,000  millionaires among Microsoft employees. Since the 1990s, it has increasingly diversified from the operating system market and has made a number of corporate acquisitions, their largest being the acquisition of LinkedIn for $26.2  billion in December 2016,[3] followed by their acquisition of Skype Technologies for $8.5 billion in May 2011.[4]
                As of 2015, Microsoft is market-dominant in the IBM PC compatible operating system market and the office software suite market, although it has lost the majority of the overall operating system market to Android.[5] The company also produces a wide range of other consumer and enterprise software for desktops, laptops, tabs, gadgets, and servers, including Internet search (with Bing), the digital services market (through MSN), mixed reality (HoloLens), cloud computing (Azure), and software development (Visual Studio).
                """;
//        var resourceDocument = """
//                There is a valet service right next door to the restaurant that is $18 cash only. Monday-Friday valet is available from 11 am to Midnight. Saturday-Sunday valet is available from 10 am-Midnight. Overnight parking is $40.  Valet is serviced by VP Parking Inc. 773.337.8999.
//                Gift cards can be purchased in person at the restaurant, or e-gift cards can be purchased on our website. Physical gift cards can be shipped at the purchaser’s expense via a traceable delivery service that will provide a shipping label and packaging upon pickup. Purchaser waives all liability from the restaurant once the gift card has been picked up.
//                we offer brunch in the main dining room every Sunday from 10am-2pm for $75 per person. Drinks are not included in that price.
//                """;


        QAInput input = new QAInput(question, resourceDocument);
        var predict = bertZooModel.predict(input);
        return predict.orElse("Not able predict");
    }

    @GET
    @NonBlocking
    @Produces(MediaType.TEXT_PLAIN)
    @Path("/model")
    public Uni<String> makeModel() {
//        return Uni.createFrom()
//                .item(makeModel.run())
//                .ifNoItem()
//                .after(Duration.ofSeconds(20))
//                .fail();
//        return Uni.createFrom()
//                .item("Running on background")
//                .invoke(makeModel::run)
//                .runSubscriptionOn(Infrastructure.getDefaultExecutor());
        eventBus.<Boolean>requestAndForget("some-address", "my payload");
        return Uni.createFrom().item("Training starts").ifNoItem().after(Duration.ofMillis(100)).recoverWithItem(() -> "fallback");
    }


    @GET
    @Produces(MediaType.TEXT_PLAIN)
    @Path("/predict")
    public Uni<String> prediction(@QueryParam("q") String question) {
        return Uni.createFrom().item(Unchecked.supplier(() -> predictService.getPrediction(question)));
    }

    @GET
    @Produces(MediaType.APPLICATION_JSON)
    @Path("/list")
    public Multi<String> listOfModel() {
        return Multi.createFrom().items(() ->  makeModel.listOfModel().stream());
    }

    //    @Blocking // Will be called on a worker thread
    @RunOnVirtualThread
    @ConsumeEvent("some-address")
    public boolean executeQuery(String payload) {
        trainModel.trainModel();
        return true;
    }

}