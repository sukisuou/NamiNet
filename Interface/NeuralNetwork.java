//holds the network     ~template~

import java.io.Serializable;

class NeuralNetwork implements Serializable{
    private static final long serialVersionUID = 1L;
    
    private NeuronLayer[] layers;
    private double[] dropoutRates;
    
    //constructor
    public NeuralNetwork(int[] layerSizes, double[] dropoutRates){
        this.dropoutRates = dropoutRates;
        layers = new NeuronLayer[layerSizes.length - 1];
        for(int i=0; i<layers.length; i++){
            boolean isOutputLayer = (i == layers.length - 1);
            layers[i] = new NeuronLayer(layerSizes[i], layerSizes[i+1], isOutputLayer);
        }
    }

    public double[] forward(double[] input){        //forwarding
        double[] a = input;
        for(int i=0; i<layers.length; i++){
            boolean isOutputLayer = (i == layers.length - 1);   //last is output
            double dropoutRate = dropoutRates[i];
            a = layers[i].forward(a, isOutputLayer, dropoutRate);    //softmax on output, leaky ReLU on others
        }
        return a;
    }

    public void train(double[] input, double[] target, double lr){      //backpropagation
        double[] yp = forward(input);

        //compute dA for output layer
        double[] dA = Functions.crossEntropyLossDerivatives(yp, target);

        //backpropagate through layers in reverse
        for(int i=layers.length-1; i>=0; i--){
            boolean isOutputLayer = (i == layers.length - 1);
            dA = layers[i].backward(dA, lr, isOutputLayer);
        }
    }

    public double[] predict(double[] input){        //predicting (after training)
        double[] a = input;
        for(int i=0; i<layers.length; i++){
            boolean isOutputLayer = (i == layers.length - 1);
            double dropoutRate = 0.0;     //no dropouts in prediction
            a = layers[i].forward(a, isOutputLayer, dropoutRate);
        }
        return a;
    }
}