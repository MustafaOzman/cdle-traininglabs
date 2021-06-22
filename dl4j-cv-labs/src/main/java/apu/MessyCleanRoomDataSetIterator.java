package apu;

import ai.certifai.Helper;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.mapdb.serializer.SerializerArrayTuple;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Random;

public class MessyCleanRoomDataSetIterator {

    // DataSetIteroator
    // make Iterator
//    Train Iterator
    // Test
    //setup
    //downloading
    //unzip
    private static int height = 224;
    private static int width = 224;
    private static  int nchannels = 3;
    private static  int numClasses = 5;
    private static double trainPerc = 0.7;
    //Images are of format given by allowedExtension
    private static  String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

    //Random number generator
    private static  Random rng  = new Random(123);

    private static String dataDir;
    private static String downloadLink;

    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static InputSplit trainData,testData;
    private static int batchSizeA;

    //scale input to 0 - 1
    private static DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    private static ImageTransform imgtransform;
    private  static  BalancedPathFilter balancedPathFilter = new BalancedPathFilter(rng,allowedExtensions,labelMaker);

    public void setup(File inputFile, int channel, int nClass, ImageTransform imageTransform, int batchSize, double trainTestRatio) throws IOException {
//        dataDir = Paths.get(
//                System.getProperty("user.home"),
//                Helper.getPropValues("dl4j_home.data")
//        ).toString();
//        downloadLink = Helper.getPropValues("dataset.dogbreed.url");
//
//        File parentDir = new File(Paths.get(dataDir,"dog-breed-identification").toString());
//
//        if(!parentDir.exists()){
//            downloadAndUnzip();
//        }
        nchannels = channel;
        batchSizeA=batchSize;
        numClasses=nClass;
        imgtransform= imageTransform;
        trainPerc = trainTestRatio;
        //1. Load the dataset

        System.out.println(inputFile);

        FileSplit parentDir = new FileSplit(inputFile,allowedExtensions,rng);

//        if(trainPerc>1){
//            throw new IllegalArgumentException("Train Percentage must be lower than 100");
//        }




        //2. Parent Path Label
//        ParentPathLabelGenerator labelGenerator = new ParentPathLabelGenerator();
//        BalancedPathFilter bPF = new BalancedPathFilter(rng, allowedExtensions, labelGenerator);

        //Files in directories under the parent dir that have "allowed extensions" split needs a random number generator for reproducibility when splitting the files into train and test
//        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, rng);

        //The balanced path filter gives you fine tune control of the min/max cases to load for each class
//        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
        if (trainPerc >= 1) {
            throw new IllegalArgumentException("Percentage of data set aside for training has to be less than 100%. Test percentage = 100 - training percentage, has to be greater than 0");
        }

        //Split the image files into train and test
        InputSplit[] filesInDirSplit = parentDir.sample(balancedPathFilter, trainPerc, 1-trainPerc);
        trainData = filesInDirSplit[0];
        testData = filesInDirSplit[1];
    }


    public static DataSetIterator makeIterator(InputSplit split, boolean training) throws IOException{
        ImageRecordReader recordReader = new ImageRecordReader(height,width,nchannels,labelMaker);
        if (training && imgtransform != null){
            recordReader.initialize(split,imgtransform);
        }else{
            recordReader.initialize(split);
        }
        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSizeA, 1, numClasses);
        DataNormalization scaler = new ImagePreProcessingScaler();
        iter.setPreProcessor(scaler);


        return iter;
    }

    public DataSetIterator trainIterator() throws IOException {
        return makeIterator(trainData, true);
    }
    public DataSetIterator testIterator() throws IOException {
        return makeIterator(testData, false);
    }


    public MessyCleanRoomDataSetIterator() throws IOException {
    }

}
