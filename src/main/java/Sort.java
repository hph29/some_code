/**
 * Created by Hangg on 2018/7/15.
 */
public class Sort {

    // split array into half recursively
    // merge sub arrays by 1. copy sub array into two new arrays, check the sub arrays and re-order origin array
    // end condition: low >= high
    static void mergeSort(int[] arr, int low, int high){

        if (low < high){
            int mid = (low + high) / 2;
            mergeSort(arr, low, mid);
            mergeSort(arr, mid + 1, high);
            merge(arr, low, mid, high);
        }
    }

    private static void merge(int[] arr, int low, int mid, int high){
        int left_size = mid - low + 1;
        int right_size = high - mid;
        int[] left = new int[left_size];
        int[] right = new int[right_size];

        for (int i = 0; i < left_size; i++){
            left[i] = arr[low + i];
        }
        for (int z = 0; z < right_size; z++){
            right[z] = arr[mid + z + 1];
        }
        int j = 0;
        int k = 0;
        while(j < left_size && k < right_size){
            if (left[j] < right[k]){
                arr[low++] = left[j++];
            }
            else {
                arr[low++] = right[k++];
            }
        }
        while(j < left_size){
            arr[low++] = left[j++];
        }
        while(k < right_size){
            arr[low++] = right[k++];
        }
    }

    // pivot the last element, partition based on the pivot point so that any number smaller than pivot on left,
    // any number bigger than pivot on right, and place the pivot on right place, recursively sort the sub arrays.
    // end condition: low >= high
    static void quickSort(int[] arr, int low, int high){
        if (low < high){
            int index = partition(arr, low, high);

            quickSort(arr, low, index -1);
            quickSort(arr, index + 1, high);
        }
    }

    private static int partition(int[] arr, int low, int high){
        int pivot = arr[high];
        int i = low - 1;
        for (int j = low;  j < high; j++){
            if (arr[j] < pivot){
                i++;
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        int temp = arr[i+1];
        arr[i+1] = arr[high];
        arr[high] = temp;

        return i + 1;
    }

    static void insertionSort(int[] arr){
        for (int i = 0; i < arr.length - 1; i++){
            for (int j = i + 1; j < arr.length; j++){
                if (arr[i] > arr[j]){
                    int temp = arr[i];
                    arr[i] = arr[j];
                    arr[j] = temp;
                }
            }
        }
    }

    static void bubbleSort(int[] arr){
        for (int i = 0; i < arr.length - 1; i++){
            for (int j = i; j < arr.length -1; j++){
                int k = j + 1;
                if (arr[j] > arr[k]){
                    int temp = arr[j];
                    arr[j] = arr[k];
                    arr[k] = temp;
                }
            }
        }
    }
}
