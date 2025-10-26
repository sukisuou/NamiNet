//a layer of neurons        ~template~

import java.util.Random;
import java.io.Serializable;

class NeuronLayer implements Serializable{
    private static final long serialVersionUID = 1L;

    private int inputSize;  //size of input (previous layer's output)
    private int outputSize; //size of output (next layer's input)
    private double[][] w;   //weight
    private double[] b;     //bias
    private double[] lastInput;
    private double[] lastZ;     //previous z
    private double[] lastA;     //previous a
    private boolean[] dropoutMask;
    private double[][] mW;
    private double[][] vW;
    private double[] mB;
    private double[] vB;
    private int t;

    //constructor
    public NeuronLayer(int inputSize, int outputSize, boolean isOutputLayer){
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.w = new double[outputSize][inputSize];
        this.b = new double[outputSize];
        initializeParameters(isOutputLayer);
    }

    public void initializeParameters(boolean isOutputLayer){
        Random rand = new Random();
        mW = new double[outputSize][inputSize]; //Adam parameters
        vW = new double[outputSize][inputSize];
        mB = new double[outputSize];
        vB = new double[outputSize];
        t = 0;

        for(int i=0; i<outputSize; i++){        //initialize w and b
            for(int j=0; j<inputSize; j++){
                if(isOutputLayer){  //Xavier initialization
                    w[i][j] = rand.nextGaussian() * Math.sqrt(1.0 / inputSize);
                }else{  //He initialization (leakyReLU)
                    w[i][j] = rand.nextGaussian() * Math.sqrt(2.0 / inputSize);
                }
            }
            b[i] = 0.0;
        }
    }

    public double[] forward(double[] input, boolean useSoftmax, double dropoutRate){    //forward pass
        double[] z = new double[outputSize];
        for(int i=0; i<outputSize; i++){
            double sum = b[i];
            for(int j=0; j<inputSize; j++){
                sum += w[i][j] * input[j];
            }
            z[i] = sum;
        }

        this.lastInput = input;
        this.lastZ = z;

        if(useSoftmax){
            this.lastA =  Functions.softmax(z);     //for output
        }else{
            this.lastA = new double[outputSize];    //for hidden layers
            for(int i=0; i<outputSize; i++){
                this.lastA[i] = Functions.leakyReLU(z[i]);
            }
        }
        
        //neuron dropouts
        if(dropoutRate > 0){
            Random rand = new Random();
            dropoutMask = new boolean[outputSize];
            for(int i=0; i<outputSize; i++){
                if(rand.nextDouble() < dropoutRate){
                    //drop the neuron
                    lastA[i] = 0.0;
                    dropoutMask[i] = false;
                }else{
                    //keep and scale to keep average consistent
                    lastA[i] /= (1.0 - dropoutRate);
                    dropoutMask[i] = true;
                }
            }
        }else{
            dropoutMask = null;
        }
        return lastA;
    }

    public double[] backward(double[] dA, double lr, boolean isOutputLayer){    //backpropagate
        double[] dZ = new double[outputSize];
        if(isOutputLayer){  //softmax (yp-yt)
            dZ = dA;
        }else{              //leaky ReLU
            for(int i=0; i<outputSize; i++){
                dZ[i] = dA[i] * Functions.leakyReLUDerivatives(lastZ[i]);
            }
        }

        //drops the gradient to 0
        if (dropoutMask != null) {
            for (int i = 0; i < dZ.length; i++) {
                if (!dropoutMask[i]) {
                    dZ[i] = 0.0;
                }
            }
        }

        //gradient for weights and bias
        double[][] dW = new double[outputSize][inputSize];
        double[] dB = new double[outputSize];
        for(int i=0; i<outputSize; i++){
            dB[i] = dZ[i];
            for(int j=0; j<inputSize; j++){
                dW[i][j] = dZ[i] * lastInput[j];
            }
        }

        //Adam optimizer
        t++;                    //time increment 
        double beta1 = 0.99;     //hyperparameters
        double beta2 = 0.999;
        double epsilon = 1e-8;

        //update weights
        double weightDecay = 0.00005;
        for(int i=0; i<outputSize; i++){
            for(int j=0; j<inputSize; j++){
                //compute m and v
                double grad = dW[i][j] + (weightDecay * w[i][j]);
                double clipValue = 5.0;         //clipping to between -5 to 5
                grad = Math.max(-clipValue, Math.min(clipValue, grad));
                double noise = (Math.random() - 0.5) * 2e-6;    //noise for plateauing problem

                mW[i][j] = beta1 * mW[i][j] + (1 - beta1) * grad;       //m and v moments
                vW[i][j] = beta2 * vW[i][j] + (1 - beta2) * Math.pow(grad, 2);

                //bias correction
                double mHat = mW[i][j] / (1 - Math.pow(beta1, t));
                double vHat = vW[i][j] / (1 - Math.pow(beta2, t));

                //update weight
                w[i][j] -= lr * mHat / (Math.sqrt(vHat) + epsilon) + noise;
            }
        }

        //update biases
        for(int i=0; i<outputSize; i++){
            //compute m and v
                mB[i] = beta1 * mB[i] + (1 - beta1) * dB[i];
                vB[i] = beta2 * vB[i] + (1 - beta2) * Math.pow(dB[i], 2);

                //bias correction
                double mHat = mB[i] / (1 - Math.pow(beta1, t));
                double vHat = vB[i] / (1 - Math.pow(beta2, t));

                //update weight
                b[i] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
        }

        //gradient for previous layer
        double[] dAprev = new double[inputSize];
        for(int j=0; j<inputSize; j++){
            double sum = 0.0;
            for(int i=0; i<outputSize; i++){
                sum +=  dZ[i] * w[i][j];
            }
            dAprev[j] = sum;
        }

        return dAprev;
    }
}