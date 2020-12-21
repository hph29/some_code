package CodingQuestions;

// Google question
// Given an array = [2,5,1,2,3,5,1,2,4]
// It should return 2

import java.util.HashSet;
import java.util.Optional;

// Given an array = [2,1,1,2,3,5,1,2,4]
// It should return 1
public class FirstRecurringChars {
    public static Optional<Integer> firstRecurringChars(int[] array){
        HashSet<Integer> set = new HashSet<>();
        for (int i =0; i<array.length; i++){
            if (!set.contains(array[i])){
                set.add(array[i]);
            }
            else{
                return Optional.of(array[i]);
            }
        }
        return Optional.empty();
    }

    public static void main(String[] args){
        System.out.println(firstRecurringChars(new int[]{2,5,1,2,3,5,1,2,4}).get());
        System.out.println(firstRecurringChars(new int[]{2,1,1,2,3,5,1,2,4}).get());
    }
}
