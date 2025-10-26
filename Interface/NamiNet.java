//main class

public class NamiNet{
    public static void main(String[] args){
        NeuralNetwork loadedModel = ModelSaver.loadModel("naminet_model.bin");
        new NamiNetGUI(loadedModel);
    }
}