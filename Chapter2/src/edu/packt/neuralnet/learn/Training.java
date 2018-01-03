package edu.packt.neuralnet.learn;

import java.util.ArrayList;

import edu.packt.neuralnet.InputLayer;
import edu.packt.neuralnet.NeuralNet;
import edu.packt.neuralnet.Neuron;

/**
 * ClassName: Training <br/>
 * Function: <br/>
 * date: 2018年1月3日 下午9:20:03 <br/>
 *
 * @author xushjie
 * @version
 * @since JDK 1.8
 */
public abstract class Training {

    private int    epochs;
    private double error;
    private double mse;

    public enum TrainingTypesENUM {
        PERCEPTRON,
        ADALINE;
    }

    /**
     * train: <br/>
     *
     * @author xushjie
     * @param n
     * @return
     * @since JDK 1.8
     */
    public NeuralNet train(NeuralNet n) {

        ArrayList<Double> inputWeightIn = new ArrayList<Double>();

        int rows = n.getTrainSet().length;
        int cols = n.getTrainSet()[0].length;

        while (this.getEpochs() < n.getMaxEpochs()) {

            double estimatedOutput = 0.0;
            double realOutput = 0.0;

            for (int i = 0; i < rows; i++) {

                double netValue = 0.0;

                for (int j = 0; j < cols; j++) {
                    inputWeightIn = n.getInputLayer()
                                     .getListOfNeurons()
                                     .get(j)
                                     .getListOfWeightIn();
                    double inputWeight = inputWeightIn.get(0);
                    netValue = netValue + inputWeight * n.getTrainSet()[i][j];
                }

                estimatedOutput = this.activationFnc(n.getActivationFnc(),
                                                     netValue);
                realOutput = n.getRealOutputSet()[i];

                this.setError(realOutput - estimatedOutput);

                // System.out.println("Epoch: "+this.getEpochs()+" / Error: " + this.getError());

                if (Math.abs(this.getError()) > n.getTargetError()) {
                    // fix weights
                    InputLayer inputLayer = new InputLayer();
                    inputLayer.setListOfNeurons(this.teachNeuronsOfLayer(cols,
                                                                         i,
                                                                         n,
                                                                         netValue));
                    n.setInputLayer(inputLayer);
                }

            }

            this.setMse(Math.pow(realOutput - estimatedOutput,
                                 2.0));
            n.getListOfMSE()
             .add(this.getMse());

            this.setEpochs(this.getEpochs() + 1);

        }

        n.setTrainingError(this.getError());

        return n;
    }

    /**
     * teachNeuronsOfLayer: <br/>
     *
     * @author xushjie
     * @param numberOfInputNeurons
     * @param line
     * @param n
     * @param netValue
     * @return
     * @since JDK 1.8
     */
    private ArrayList<Neuron> teachNeuronsOfLayer(int numberOfInputNeurons,
                                                  int line,
                                                  NeuralNet n,
                                                  double netValue) {
        ArrayList<Neuron> listOfNeurons = new ArrayList<Neuron>();
        ArrayList<Double> inputWeightsInNew = new ArrayList<Double>();
        ArrayList<Double> inputWeightsInOld = new ArrayList<Double>();

        for (int j = 0; j < numberOfInputNeurons; j++) {
            inputWeightsInOld = n.getInputLayer()
                                 .getListOfNeurons()
                                 .get(j)
                                 .getListOfWeightIn();
            double inputWeightOld = inputWeightsInOld.get(0);

            inputWeightsInNew.add(this.calcNewWeight(n.getTrainType(),
                                                     inputWeightOld,
                                                     n,
                                                     this.getError(),
                                                     n.getTrainSet()[line][j],
                                                     netValue));

            Neuron neuron = new Neuron();
            neuron.setListOfWeightIn(inputWeightsInNew);
            listOfNeurons.add(neuron);
            inputWeightsInNew = new ArrayList<Double>();
        }

        return listOfNeurons;

    }

    /**
     * calcNewWeight: <br/>
     *
     * @author xushjie
     * @param trainType
     * @param inputWeightOld
     * @param n
     * @param error
     * @param trainSample
     * @param netValue
     * @return
     * @since JDK 1.8
     */
    private double calcNewWeight(TrainingTypesENUM trainType,
                                 double inputWeightOld,
                                 NeuralNet n,
                                 double error,
                                 double trainSample,
                                 double netValue) {
        switch (trainType) {
            case PERCEPTRON:
                return inputWeightOld + n.getLearningRate() * error * trainSample;
            case ADALINE:
                return inputWeightOld + n.getLearningRate() * error * trainSample * derivativeActivationFnc(n.getActivationFnc(),
                                                                                                            netValue);
            default:
                throw new IllegalArgumentException(trainType + " does not exist in TrainingTypesENUM");
        }
    }

    /**
     * ClassName: ActivationFncENUM <br/>
     * Function: <br/>
     * date: 2018年1月3日 下午9:52:13 <br/>
     *
     * @author xushjie
     * @version Training
     * @since JDK 1.8
     */
    public enum ActivationFncENUM {
        STEP,
        LINEAR,
        SIGLOG,
        HYPERTAN;
    }

    /**
     * activationFnc: <br/>
     *
     * @author xushjie
     * @param fnc
     * @param value
     * @return
     * @since JDK 1.8
     */
    private double activationFnc(ActivationFncENUM fnc,
                                 double value) {
        switch (fnc) {
            case STEP:
                return fncStep(value);
            case LINEAR:
                return fncLinear(value);
            case SIGLOG:
                return fncSigLog(value);
            case HYPERTAN:
                return fncHyperTan(value);
            default:
                throw new IllegalArgumentException(fnc + " does not exist in ActivationFncENUM");
        }
    }

    /**
     * derivativeActivationFnc: <br/>
     *
     * @author xushjie
     * @param fnc
     * @param value
     * @return
     * @since JDK 1.8
     */
    public double derivativeActivationFnc(ActivationFncENUM fnc,
                                          double value) {
        switch (fnc) {
            case LINEAR:
                return derivativeFncLinear(value);
            case SIGLOG:
                return derivativeFncSigLog(value);
            case HYPERTAN:
                return derivativeFncHyperTan(value);
            default:
                throw new IllegalArgumentException(fnc + " does not exist in ActivationFncENUM");
        }
    }

    /**
     * fncStep: <br/>
     *
     * @author xushjie
     * @param v
     * @return
     * @since JDK 1.8
     */
    private double fncStep(double v) {
        if (v >= 0) {
            return 1.0;
        } else {
            return 0.0;
        }
    }

    /**
     * fncLinear: <br/>
     *
     * @author xushjie
     * @param v
     * @return
     * @since JDK 1.8
     */
    private double fncLinear(double v) {
        return v;
    }

    /**
     * fncSigLog: <br/>
     *
     * @author xushjie
     * @param v
     * @return
     * @since JDK 1.8
     */
    private double fncSigLog(double v) {
        return 1.0 / (1.0 + Math.exp(-v));
    }

    /**
     * fncHyperTan: <br/>
     *
     * @author xushjie
     * @param v
     * @return
     * @since JDK 1.8
     */
    private double fncHyperTan(double v) {
        return Math.tanh(v);
    }

    /**
     * derivativeFncLinear: <br/>
     *
     * @author xushjie
     * @param v
     * @return
     * @since JDK 1.8
     */
    private double derivativeFncLinear(double v) {
        return 1.0;
    }

    /**
     * derivativeFncSigLog: <br/>
     *
     * @author xushjie
     * @param v
     * @return
     * @since JDK 1.8
     */
    private double derivativeFncSigLog(double v) {
        return v * (1.0 - v);
    }

    /**
     * derivativeFncHyperTan: <br/>
     *
     * @author xushjie
     * @param v
     * @return
     * @since JDK 1.8
     */
    private double derivativeFncHyperTan(double v) {
        return (1.0 / Math.pow(Math.cosh(v),
                               2.0));
    }

    /**
     * printTrainedNetResult: <br/>
     *
     * @author xushjie
     * @param trainedNet
     * @since JDK 1.8
     */
    public void printTrainedNetResult(NeuralNet trainedNet) {

        int rows = trainedNet.getTrainSet().length;
        int cols = trainedNet.getTrainSet()[0].length;

        ArrayList<Double> inputWeightIn = new ArrayList<Double>();

        for (int i = 0; i < rows; i++) {
            double netValue = 0.0;
            for (int j = 0; j < cols; j++) {
                inputWeightIn = trainedNet.getInputLayer()
                                          .getListOfNeurons()
                                          .get(j)
                                          .getListOfWeightIn();
                double inputWeight = inputWeightIn.get(0);
                netValue = netValue + inputWeight * trainedNet.getTrainSet()[i][j];

                System.out.print(trainedNet.getTrainSet()[i][j] + "\t");
            }

            double estimatedOutput = this.activationFnc(trainedNet.getActivationFnc(),
                                                        netValue);

            System.out.print(" NET OUTPUT: " + estimatedOutput + "\t");
            System.out.print(" REAL OUTPUT: " + trainedNet.getRealOutputSet()[i] + "\t");
            double error = estimatedOutput - trainedNet.getRealOutputSet()[i];
            System.out.print(" ERROR: " + error + "\n");

        }

    }

    public int getEpochs() {
        return epochs;
    }

    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }

    public double getError() {
        return error;
    }

    public void setError(double error) {
        this.error = error;
    }

    public double getMse() {
        return mse;
    }

    public void setMse(double mse) {
        this.mse = mse;
    }

}
