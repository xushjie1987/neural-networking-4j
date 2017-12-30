package edu.packt.neuralnet;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * ClassName: HiddenLayer <br/>
 * Function: <br/>
 * date: 2017年12月29日 下午10:48:11 <br/>
 *
 * @author xushjie
 * @version
 * @since JDK 1.8
 */
public class HiddenLayer extends Layer {

    /**
     * initLayer: <br/>
     *
     * @author xushjie
     * @param hiddenLayer
     * @param listOfHiddenLayer
     * @param inputLayer
     * @param outputLayer
     * @return
     * @since JDK 1.8
     */
    public ArrayList<HiddenLayer> initLayer(HiddenLayer hiddenLayer,
                                            ArrayList<HiddenLayer> listOfHiddenLayer,
                                            InputLayer inputLayer,
                                            OutputLayer outputLayer) {

        ArrayList<Double> listOfWeightIn = new ArrayList<Double>();
        ArrayList<Double> listOfWeightOut = new ArrayList<Double>();
        ArrayList<Neuron> listOfNeurons = new ArrayList<Neuron>();

        int numberOfHiddenLayers = listOfHiddenLayer.size();

        for (int i = 0; i < numberOfHiddenLayers; i++) {
            for (int j = 0; j < hiddenLayer.getNumberOfNeuronsInLayer(); j++) {
                Neuron neuron = new Neuron();

                int limitIn;
                int limitOut;

                if (i == 0) { // first
                    limitIn = inputLayer.getNumberOfNeuronsInLayer();
                    if (numberOfHiddenLayers > 1) {
                        limitOut = listOfHiddenLayer.get(i + 1)
                                                    .getNumberOfNeuronsInLayer();
                    } else {
                        limitOut = listOfHiddenLayer.get(i)
                                                    .getNumberOfNeuronsInLayer();
                    }
                } else if (i == numberOfHiddenLayers - 1) { // last
                    limitIn = listOfHiddenLayer.get(i - 1)
                                               .getNumberOfNeuronsInLayer();
                    limitOut = outputLayer.getNumberOfNeuronsInLayer();
                } else { // middle
                    limitIn = listOfHiddenLayer.get(i - 1)
                                               .getNumberOfNeuronsInLayer();
                    limitOut = listOfHiddenLayer.get(i + 1)
                                                .getNumberOfNeuronsInLayer();
                }

                for (int k = 0; k < limitIn; k++) {
                    listOfWeightIn.add(neuron.initNeuron());
                }
                for (int k = 0; k < limitOut; k++) {
                    listOfWeightOut.add(neuron.initNeuron());
                }

                neuron.setListOfWeightIn(listOfWeightIn);
                neuron.setListOfWeightOut(listOfWeightOut);
                listOfNeurons.add(neuron);

                listOfWeightIn = new ArrayList<Double>();
                listOfWeightOut = new ArrayList<Double>();

            }

            listOfHiddenLayer.get(i)
                             .setListOfNeurons(listOfNeurons);

            listOfNeurons = new ArrayList<Neuron>();

        }

        return listOfHiddenLayer;

    }

    /**
     * printLayer: <br/>
     *
     * @author xushjie
     * @param listOfHiddenLayer
     * @since JDK 1.8
     */
    public void printLayer(ArrayList<HiddenLayer> listOfHiddenLayer) {
        System.out.println("### HIDDEN LAYER ###");
        int h = 1;
        for (HiddenLayer hiddenLayer : listOfHiddenLayer) {
            System.out.println("Hidden Layer #" + h);
            int n = 1;
            for (Neuron neuron : hiddenLayer.getListOfNeurons()) {
                System.out.println("Neuron #" + n);
                System.out.println("Input Weights:");
                System.out.println(Arrays.deepToString(neuron.getListOfWeightIn()
                                                             .toArray()));
                System.out.println("Output Weights:");
                System.out.println(Arrays.deepToString(neuron.getListOfWeightOut()
                                                             .toArray()));
                n++;
            }
            h++;
        }
    }
}
