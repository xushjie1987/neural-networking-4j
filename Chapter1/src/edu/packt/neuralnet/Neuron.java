package edu.packt.neuralnet;

import java.util.ArrayList;
import java.util.Random;

/**
 * ClassName: Neuron <br/>
 * 神经元 <br>
 * Function: <br/>
 * date: 2017年12月29日 下午10:49:03 <br/>
 *
 * @author xushjie
 * @version
 * @since JDK 1.8
 */
public class Neuron {

    private ArrayList<Double> listOfWeightIn;
    private ArrayList<Double> listOfWeightOut;

    /**
     * initNeuron: <br/>
     *
     * @author xushjie
     * @return
     * @since JDK 1.8
     */
    public double initNeuron() {
        Random r = new Random();
        return r.nextDouble();
    }

    /**
     * getListOfWeightIn: <br/>
     *
     * @author xushjie
     * @return
     * @since JDK 1.8
     */
    public ArrayList<Double> getListOfWeightIn() {
        return listOfWeightIn;
    }

    /**
     * setListOfWeightIn: <br/>
     *
     * @author xushjie
     * @param listOfWeightIn
     * @since JDK 1.8
     */
    public void setListOfWeightIn(ArrayList<Double> listOfWeightIn) {
        this.listOfWeightIn = listOfWeightIn;
    }

    /**
     * getListOfWeightOut: <br/>
     *
     * @author xushjie
     * @return
     * @since JDK 1.8
     */
    public ArrayList<Double> getListOfWeightOut() {
        return listOfWeightOut;
    }

    /**
     * setListOfWeightOut: <br/>
     *
     * @author xushjie
     * @param listOfWeightOut
     * @since JDK 1.8
     */
    public void setListOfWeightOut(ArrayList<Double> listOfWeightOut) {
        this.listOfWeightOut = listOfWeightOut;
    }

}
