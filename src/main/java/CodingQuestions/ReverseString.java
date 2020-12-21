package CodingQuestions;

public class ReverseString {

    public static String reverse(String inputString){
        StringBuilder result = new StringBuilder();
        for(int i=inputString.length()-1; i >= 0; i--){
            result.append(inputString.charAt(i));
        }
        return result.toString();
    }

    public static void main(String[] args){
           String inputString = "Hi";
           assert reverse(inputString) == "iH";

           String greetingString = "How are you?";
           assert reverse(greetingString) == "?uoy era woH";
    }
}
