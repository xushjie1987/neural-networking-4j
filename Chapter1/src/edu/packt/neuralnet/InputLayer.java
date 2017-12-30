package edu.packt.neuralnet;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * ClassName: InputLayer <br/>
 * Function: <br/>
 * date: 2017年12月29日 下午10:48:26 <br/>
 *
 * @author xushjie
 * @version
 * @since JDK 1.8
 */
public class InputLayer extends Layer {

    /**
     * initLayer: <br/>
     *
     * @author xushjie
     * @param inputLayer
     * @return
     * @since JDK 1.8
     */
    public InputLayer initLayer(InputLayer inputLayer) {

        ArrayList<Double> listOfWeightInTemp = new ArrayList<Double>();
        ArrayList<Neuron> listOfNeurons = new ArrayList<Neuron>();

        for (int i = 0; i < inputLayer.getNumberOfNeuronsInLayer(); i++) {
            Neuron neuron = new Neuron();

            listOfWeightInTemp.add(neuron.initNeuron());

            neuron.setListOfWeightIn(listOfWeightInTemp);
            listOfNeurons.add(neuron);

            listOfWeightInTemp = new ArrayList<Double>();
        }

        inputLayer.setListOfNeurons(listOfNeurons);

        return inputLayer;
    }

    /**
     * printLayer: <br/>
     *
     * @author xushjie
     * @param inputLayer
     * @since JDK 1.8
     */
    public void printLayer(InputLayer inputLayer) {
        System.out.println("### INPUT LAYER ###");
        int n = 1;
        for (Neuron neuron : inputLayer.getListOfNeurons()) {
            System.out.println("Neuron #" + n + ":");
            System.out.println("Input Weights:");
            System.out.println(Arrays.deepToString(neuron.getListOfWeightIn()
                                                         .toArray()));
            n++;
        }
    }
}
