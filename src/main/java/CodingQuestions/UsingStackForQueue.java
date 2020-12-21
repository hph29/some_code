package CodingQuestions;
import java.util.Stack;
class MyQueue {
    Stack s1;
    Stack s2;
    int front;

    /** Initialize your data structure here. */
    public MyQueue() {
        this.s1 = new Stack();
        this.s2 = new Stack();

    }

    /** Push element x to the back of queue. */
    public void push(int x) {

        this.s1.push(x);
    }

    /** Removes the element from in front of queue and returns that element. */
    public int pop() {
        if(s2.empty()){
            while(!s1.empty()){
                s2.push(s1.pop());
            }
        }
        return (int) s2.pop();
    }

    /** Get the front element. */
    public int peek() {
        if (!s2.isEmpty()) {
            return (int) s2.peek();
        } else {
            while (!s1.isEmpty())
                s2.push(s1.pop());
        }
        return (int) s2.peek();
    }

    /** Returns whether the queue is empty. */
    public boolean empty() {
        return this.s1.empty() && this.s2.empty();
    }

    public static void main(String[] args){
        MyQueue q = new MyQueue();
        q.push(1);
        System.out.println(q.peek());
        q.push(2);
        System.out.println(q.peek());
        q.push(3);
        q.push(4);
        q.push(5);
        System.out.println(q.peek());
        System.out.println(q.pop());
        System.out.println(q.pop());
        System.out.println(q.pop());
    }
}
