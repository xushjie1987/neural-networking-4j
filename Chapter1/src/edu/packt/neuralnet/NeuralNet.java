package edu.packt.neuralnet;

import java.util.ArrayList;

/**
 * ClassName: NeuralNet <br/>
 * Function: <br/>
 * date: 2017年12月29日 下午10:47:40 <br/>
 *
 * @author xushjie
 * @version
 * @since JDK 1.8
 */
public class NeuralNet {

    private InputLayer             inputLayer;
    private HiddenLayer            hiddenLayer;
    private ArrayList<HiddenLayer> listOfHiddenLayer;
    private OutputLayer            outputLayer;
    private int                    numberOfHiddenLayers;

    /**
     * initNet: <br/>
     *
     * @author xushjie
     * @since JDK 1.8
     */
    public void initNet() {
        inputLayer = new InputLayer();
        inputLayer.setNumberOfNeuronsInLayer(2);

        numberOfHiddenLayers = 2;
        listOfHiddenLayer = new ArrayList<HiddenLayer>();
        for (int i = 0; i < numberOfHiddenLayers; i++) {
            hiddenLayer = new HiddenLayer();
            hiddenLayer.setNumberOfNeuronsInLayer(3);
            listOfHiddenLayer.add(hiddenLayer);
        }

        outputLayer = new OutputLayer();
        outputLayer.setNumberOfNeuronsInLayer(1);

        inputLayer = inputLayer.initLayer(inputLayer);

        listOfHiddenLayer = hiddenLayer.initLayer(hiddenLayer,
                                                  listOfHiddenLayer,
                                                  inputLayer,
                                                  outputLayer);

        outputLayer = outputLayer.initLayer(outputLayer);

    }

    /**
     * printNet: <br/>
     *
     * @author xushjie
     * @since JDK 1.8
     */
    public void printNet() {
        inputLayer.printLayer(inputLayer);
        System.out.println();
        hiddenLayer.printLayer(listOfHiddenLayer);
        System.out.println();
        outputLayer.printLayer(outputLayer);
    }

}
