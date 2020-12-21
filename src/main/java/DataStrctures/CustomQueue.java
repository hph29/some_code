package DataStrctures;


class Node3<E> {
    Node3<E> next;
    E data;

    Node3(E value){
        this.data = value;
        this.next = null;
    }
}

public class CustomQueue<E> {
    Node3<E> first;
    Node3<E> last;
    int length;

    CustomQueue(){
        this.first = null;
        this.last = null;
        this.length = 0;
    }

    public E peek(){
        return this.first.data;
    }

    public void enqueue(E value){
        Node3<E> newNode = new Node3<>(value);
        if (this.length == 0){
            this.first = newNode;
        }
        else {
            this.last.next = newNode;
        }
        this.last = newNode;
        this.length++;

    }

    public E dequeue(){
        Node3<E> firstNode = this.first;

        if (this.length == 0){
            return null;
        }
        if (this.length == 1){
            this.last = null;
            this.first = null;
        }
        else{
            this.first = this.first.next;
        }
        this.length--;
        return firstNode.data;
    }

    @Override
    public String toString() {
        if (this.length == 0){
            return "Empty!";
        }
        else{
            return "Top: " + this.first.data.toString() + " ,BottomL: "
                    + this.last.data.toString();
        }



    }

    public static void main(String[] args){
        CustomQueue<Integer> cq = new CustomQueue<>();
        cq.enqueue(1);
        System.out.println(cq.toString());
        cq.enqueue(2);
        System.out.println(cq.toString());
        cq.enqueue(3);
        System.out.println(cq.toString());
        System.out.println(cq.dequeue());
        System.out.println(cq.toString());
        System.out.println(cq.peek());
        System.out.println(cq.toString());
        System.out.println(cq.dequeue());
        System.out.println(cq.toString());
        System.out.println(cq.dequeue());
        System.out.println(cq.toString());

    }

}
