package DataStrctures;

import java.io.PrintStream;

class Node2<E> {
    Node2<E> next;
    E data;

    Node2(E value){
        this.data = value;
        this.next = null;
    }
}

public class CustomStack<E> {
    Node2<E> top;
    Node2<E> bottom;
    int length;

    CustomStack(){
        this.top = null;
        this.bottom = null;
        this.length = 0;
    }

    public E peek(){
        return this.top.data;
    }

    public void push(E value){
        Node2<E> newNode = new Node2<>(value);
        if (this.length == 0){
            this.bottom = newNode;
        }
        newNode.next = this.top;
        this.top = newNode;
        this.length++;
    }

    public E pop(){
        Node2<E> topNode = this.top;
        if (this.length == 0){
            return null;
        }
        if (this.length == 1){
            this.bottom = null;
            this.top = null;
        }
        else{
            this.top = this.top.next;
        }
        this.length--;
        return topNode.data;
    }

    @Override
    public String toString() {
        if (this.length == 0){
            return "Empty!";
        }
        else{
            return "Top: " + this.top.data.toString() + " ,BottomL: "
                    + this.bottom.data.toString();
        }



    }

    public static void main(String[] args){
        CustomStack<Integer> cs = new CustomStack<>();
        cs.push(1);
        System.out.println(cs.toString());
        cs.push(2);
        System.out.println(cs.toString());
        cs.push(3);
        System.out.println(cs.toString());
        System.out.println(cs.pop());
        System.out.println(cs.toString());
        System.out.println(cs.peek());
        System.out.println(cs.toString());
        System.out.println(cs.pop());
        System.out.println(cs.toString());
        System.out.println(cs.pop());
        System.out.println(cs.toString());

    }

}
