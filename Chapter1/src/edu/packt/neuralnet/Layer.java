package edu.packt.neuralnet;

import java.util.ArrayList;

/**
 * ClassName: Layer <br/>
 * Function: <br/>
 * date: 2017年12月29日 下午10:48:35 <br/>
 *
 * @author xushjie
 * @version
 * @since JDK 1.8
 */
public abstract class Layer {

    private ArrayList<Neuron> listOfNeurons;
    private int               numberOfNeuronsInLayer;

    /**
     * printLayer: <br/>
     *
     * @author xushjie
     * @since JDK 1.8
     */
    public void printLayer() {
    }

    /**
     * getListOfNeurons: <br/>
     *
     * @author xushjie
     * @return
     * @since JDK 1.8
     */
    public ArrayList<Neuron> getListOfNeurons() {
        return listOfNeurons;
    }

    /**
     * setListOfNeurons: <br/>
     *
     * @author xushjie
     * @param listOfNeurons
     * @since JDK 1.8
     */
    public void setListOfNeurons(ArrayList<Neuron> listOfNeurons) {
        this.listOfNeurons = listOfNeurons;
    }

    /**
     * getNumberOfNeuronsInLayer: <br/>
     *
     * @author xushjie
     * @return
     * @since JDK 1.8
     */
    public int getNumberOfNeuronsInLayer() {
        return numberOfNeuronsInLayer;
    }

    /**
     * setNumberOfNeuronsInLayer: <br/>
     *
     * @author xushjie
     * @param numberOfNeuronsInLayer
     * @since JDK 1.8
     */
    public void setNumberOfNeuronsInLayer(int numberOfNeuronsInLayer) {
        this.numberOfNeuronsInLayer = numberOfNeuronsInLayer;
    }

}
