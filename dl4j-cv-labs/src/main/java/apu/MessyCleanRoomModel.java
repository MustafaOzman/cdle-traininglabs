package apu;

import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.transform.*;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class MessyCleanRoomModel {
    // Messy clean room classification
    // call the dataset Iterator class and the methods
    // Image Transformation
    private static int seed = 123;
    private static Random rng = new Random(seed);
    private static final String[] allowedFormats = BaseImageLoader.ALLOWED_FORMATS;
    private static double trainPerc = 0.7;
    private static double lr = 0.001;
    private static int height = 50;
    private static int width = 50;
    private static int batchSize = 50;
    private static int nchannel = 3;
    private static int nClasses = 3;
    private static int nEpoch = 10;
    private static int nOutput = 50;

    public static void main(String[] args) throws IOException {

        File inputFile = new ClassPathResource("messy-vs-clean-room-dataset/train").getFile();
//        MessyCleanRoomDataSetIterator.setup(inputFile, channel, nclass,imagetransofrmation);
        // Image Transform
        ImageTransform HFlip = new FlipImageTransform(1);
        ImageTransform rCrop = new RandomCropTransform(seed,50,50);
        ImageTransform rotate = new RotateImageTransform(5);

        List<Pair<ImageTransform,Double>> pipeline= Arrays.asList(
                new Pair<>(HFlip,0.3),
                new Pair<>(rCrop,0.4),
                new Pair<>(rotate,0.2)
        );

        ImageTransform tp = new PipelineImageTransform( pipeline,false);


        MessyCleanRoomDataSetIterator roomIterator= new MessyCleanRoomDataSetIterator();
        roomIterator.setup(inputFile,nchannel,nOutput,tp,batchSize,0.7);

        DataSetIterator trainIter = roomIterator.trainIterator();
        DataSetIterator testIter = roomIterator.testIterator();



        MultiLayerConfiguration mcrConfig = new NeuralNetConfiguration.Builder()
                //hyperParameters
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(lr))

                .list()
                .layer(0, new ConvolutionLayer.Builder()    //index 0 refers to the number of the layer
                        .nIn(nchannel)         //number of channels from the input image
                        .nOut(100)              //number of kernels / filters to be used in this convolution process
                        .kernelSize(1,1)        //he = 100. wid = 100, depth = 100
                        .stride(2,2)
                        .padding(1,1)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new ConvolutionLayer.Builder()    //index 0 refers to the number of the layer
                        .nOut(100)              //number of kernels / filters to be used in this convolution process
                        .kernelSize(5,5)        //he = 100. wid = 100, depth = 100
                        .stride(2,2)
                        .padding(1,1)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new ConvolutionLayer.Builder()    //index 0 refers to the number of the layer
                        .nOut(100)              //number of kernels / filters to be used in this convolution process
                        .kernelSize(5,5)        //he = 100. wid = 100, depth = 100
                        .stride(2,2)
                        .padding(1,1)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new SubsamplingLayer.Builder()            //Pooling layer
                        .kernelSize(3,3)                                //The pooling filter
                        .stride(2,2)                                    //Stride - which decides the downsampling
                        .poolingType(SubsamplingLayer.PoolingType.MAX)  //pooling type
                        .build())
                .layer(4, new ConvolutionLayer.Builder()    //index 0 refers to the number of the layer
                        .nOut(100)              //number of kernels / filters to be used in this convolution process
                        .kernelSize(3,3)        //he = 100. wid = 100, depth = 100
                        .stride(2,2)
                        .padding(1,1)
                        .activation(Activation.RELU)
                        .build())
                .layer(5, new SubsamplingLayer.Builder()            //Pooling layer
                        .kernelSize(3,3)                                //The pooling filter
                        .stride(2,2)                                    //Stride - which decides the downsampling
                        .poolingType(SubsamplingLayer.PoolingType.MAX)  //pooling type
                        .build())
                .layer(6, new DenseLayer.Builder()                      //Fully connected layer
                        .nOut(50)                                           //number of neurons
                        .activation(Activation.ELU)
                        .build())
                .layer(7, new DenseLayer.Builder()
                        .nOut(50)
                        .activation(Activation.ELU)
                        .build())
                .layer(8, new OutputLayer.Builder()
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(nClasses)
                        .build())
                .setInputType(InputType.convolutional(height,width,nchannel))
                .build();


        //7. Model Training
        MultiLayerNetwork rpsModel = new MultiLayerNetwork(mcrConfig);
        rpsModel.init();

        System.out.println("rpsModel"+rpsModel.summary());


        //8. Evaluation
        Evaluation evalTrain = rpsModel.evaluate(trainIter);
        Evaluation evalTest = rpsModel.evaluate(testIter);

        System.out.println("Training Evaluation Results: "+ evalTrain.stats());
        System.out.println("Test Evaluation Results: "+ evalTest.stats());

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        rpsModel.setListeners(
                new StatsListener(statsStorage),
                new ScoreIterationListener(10),
                new EvaluativeListener(trainIter, 1, InvocationType.EPOCH_END),
                new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END)
        );

        //model Fitting
        rpsModel.fit(trainIter,nEpoch);

    }
}
