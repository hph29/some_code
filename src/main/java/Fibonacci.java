/**
 * Created by Hangg on 2018/7/17.
 */
public class Fibonacci {

    // time complexity O(n-1) + O(n-2) exponential
    // space O(n) for stack, or O(1) if not consider stack
    static int fibonacci1(int n){
        if (n <= 1) {
            return n;
        }
        else {
            return fibonacci1(n - 1) + fibonacci1(n - 2);
        }
    }

    // time O(n)
    // space O(n)
    static int fibonacci2(int n){
        int[] arr = new int[n+2];
        arr[0] = 0;
        arr[1] = 1;

        for(int i=2; i<=n; i++){
            arr[i] = arr[i-1] + arr[i-2];
        }

        return arr[n];
    }

    // time O(n)
    // space O(1)
    static int fibonacci3(int n){

        int a = 0;
        int b = 1;
        int c = a + b;
        if (n <= 1){
            return n;
        }

        for(int i = 3; i<=n; i++){
            a = b;
            b = c;
            c = a + b;
        }

        return c;
    }
}
