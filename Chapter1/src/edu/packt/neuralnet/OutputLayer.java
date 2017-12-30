package edu.packt.neuralnet;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * ClassName: OutputLayer <br/>
 * Function: <br/>
 * date: 2017年12月29日 下午10:47:58 <br/>
 *
 * @author xushjie
 * @version
 * @since JDK 1.8
 */
public class OutputLayer extends Layer {

    /**
     * initLayer: <br/>
     *
     * @author xushjie
     * @param outputLayer
     * @return
     * @since JDK 1.8
     */
    public OutputLayer initLayer(OutputLayer outputLayer) {
        ArrayList<Double> listOfWeightOutTemp = new ArrayList<Double>();
        ArrayList<Neuron> listOfNeurons = new ArrayList<Neuron>();

        for (int i = 0; i < outputLayer.getNumberOfNeuronsInLayer(); i++) {
            Neuron neuron = new Neuron();

            listOfWeightOutTemp.add(neuron.initNeuron());

            neuron.setListOfWeightOut(listOfWeightOutTemp);
            listOfNeurons.add(neuron);

            listOfWeightOutTemp = new ArrayList<Double>();
        }

        outputLayer.setListOfNeurons(listOfNeurons);

        return outputLayer;

    }

    /**
     * printLayer: <br/>
     *
     * @author xushjie
     * @param outputLayer
     * @since JDK 1.8
     */
    public void printLayer(OutputLayer outputLayer) {
        System.out.println("### OUTPUT LAYER ###");
        int n = 1;
        for (Neuron neuron : outputLayer.getListOfNeurons()) {
            System.out.println("Neuron #" + n + ":");
            System.out.println("Output Weights:");
            System.out.println(Arrays.deepToString(neuron.getListOfWeightOut()
                                                         .toArray()));
            n++;
        }
    }

}
