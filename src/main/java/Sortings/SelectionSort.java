package Sortings;

import java.util.Arrays;

public class SelectionSort {

    static void swap(int[] arr, int i, int j){
        System.out.println("swapping" + " " + arr[i] + " " + arr[j]);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }

    static int[] selectionSort(int[] arr){

        for(int i = 0; i < arr.length -1 ; i++){
            int min = arr[i];
            int minIndex = i;
            for (int j = i+1; j < arr.length; j++){
                if(arr[j] < min){
                    min = arr[j];
                    minIndex = j;
                }
            }
            swap(arr, i, minIndex);

        }
        return arr;
    }

    public static void main(String[] args){
      int[] arr = new int[] {9,8,7,1,2,3,6,5,4};
      int[] newArr = selectionSort(arr);
      Arrays.stream(newArr).forEach(value -> System.out.print(value + " "));
    }
}
