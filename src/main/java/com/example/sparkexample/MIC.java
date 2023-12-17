package com.example.sparkexample;
import org.apache.spark.sql.functions;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;

import org.apache.spark.api.java.*;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.optimization.*;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import java.util.ArrayList;
import java.util.List;

public final class MIC {
    public static void main(String[] args) {

        final double a1 = 980.63;
        final double a2 = 0.70;
        final double a3 = 0.58;
        final double a4 = 0.58;
        final double b1 = 0.86;

        double pov = 0.15;
        double size = 0.3;
        double sigma = 0.12;

        SparkSession spark = SparkSession
                .builder()
                .appName("JavaSparkPi")
                .config("spark.master", "local")
                .getOrCreate();

        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());

        int slices = 2;
        int n = 10000 * slices;
        List<Integer> l = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            l.add(i);
        }

        double instant_mic = a1*Math.pow(size,a2)*Math.pow(sigma,a3);
        double mic=b1*instant_mic*Math.pow(pov,a4)+(1-b1)*instant_mic;

        JavaRDD<Integer> dataSet = jsc.parallelize(l, slices);

        int count = dataSet.map(integer -> {
            double x = Math.random() * 2 - 1;
            double y = Math.random() * 2 - 1;
            return (x * x + y * y <= 1) ? 1 : 0;
        }).reduce(Integer::sum);

        System.out.println("Mic is roughly " + mic);
        Calibrate();

        spark.stop();
    }

    private double Calibrate() {

        //Calibrate MIC model

        String path = "sample_libsvm_data.txt";
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc, path).toJavaRDD();
        int numFeatures = data.take(1).get(0).features().size();

// Split initial RDD into two... [60% training data, 40% testing data].
        JavaRDD<LabeledPoint> trainingInit = data.sample(false, 0.6, 11L);
        JavaRDD<LabeledPoint> test = data.subtract(trainingInit);

// Append 1 into the training data as intercept.
        JavaPairRDD<Object, Vector> training = data.mapToPair(p ->
                new Tuple2<>(p.label(), MLUtils.appendBias(p.features())));
        training.cache();

// Run training algorithm to build the model.
        int numCorrections = 10;
        double convergenceTol = 1e-4;
        int maxNumIterations = 20;
        double regParam = 0.1;
        Vector initialWeightsWithIntercept = Vectors.dense(new double[numFeatures + 1]);

        Tuple2<Vector, double[]> result = LBFGS.runLBFGS(
                training.rdd(),
                new LogisticGradient(),
                new SquaredL2Updater(),
                numCorrections,
                convergenceTol,
                maxNumIterations,
                regParam,
                initialWeightsWithIntercept);
        Vector weightsWithIntercept = result._1();
        double[] loss = result._2();

        LogisticRegressionModel model = new LogisticRegressionModel(
                Vectors.dense(Arrays.copyOf(weightsWithIntercept.toArray(), weightsWithIntercept.size() - 1)),
                (weightsWithIntercept.toArray())[weightsWithIntercept.size() - 1]);

// Clear the default threshold.
        model.clearThreshold();

// Compute raw scores on the test set.
        JavaPairRDD<Object, Object> scoreAndLabels = test.mapToPair(p ->
                new Tuple2<>(model.predict(p.features()), p.label()));

// Get evaluation metrics.
        BinaryClassificationMetrics metrics =
                new BinaryClassificationMetrics(scoreAndLabels.rdd());
        double auROC = metrics.areaUnderROC();

        System.out.println("Loss of each step in training process");
        for (double l : loss) {
            System.out.println(l);
        }
        System.out.println("Area under ROC = " + auROC);
    }

}

