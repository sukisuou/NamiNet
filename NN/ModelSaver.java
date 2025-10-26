//to save trained data  

import java.io.*;

public class ModelSaver{
    public static void saveModel(NeuralNetwork net, String filename){
        try(ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename))){
            oos.writeObject(net);
            System.out.println("Model saved to: " + filename);
        }catch(Exception e){
            e.printStackTrace();
        }
    }

    public static NeuralNetwork loadModel(String filename){
        try(ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))){
            NeuralNetwork net = (NeuralNetwork) ois.readObject();
            System.out.println("Model loaded from: " + filename);
            return net;
        }catch(Exception e){
            e.printStackTrace();
            return null;
        }
    }
}