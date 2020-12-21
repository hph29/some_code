package DataStrctures;

class Node<E>{
    E data;
    Node next;
    Node(E data){
        this.data = data;
        this.next = null;
    }
}

public class CustomLinkedList<E> {
    Node<E> head;
    Node<E> tail;
    int size;

    public void add(E e){
        if (this.size == 0){
            head = new Node<>(e);
            tail = head;

        }
        else{
            tail.next = new Node<>(e);
            tail = tail.next;
        }
        size++;
    }

    public void prepand(E e){
        if (this.size == 0){
            this.head = new Node<>(e);
            this.tail = head;
        }
        else{
            Node<E> node = new Node<>(e);
            node.next = head;
            this.head = node;
        }
        size++;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        if (this.head != null){
            Node cur = this.head;
            while (cur != null){
                sb.append(cur.data);
                sb.append(" ");
                cur = cur.next;
            }
        }
        return sb.toString();
    }

    public Node<E> traverseToIndex(int index){
        int currentIndex = 0;
        Node<E> currentNode = this.head;
        while(currentNode != null && currentIndex != index){
            currentIndex++;
            currentNode = currentNode.next;
        }
        return currentNode;
    }

    public void insert(int index, E value){
        Node<E> node = new Node<>(value);
        if (this.size == 0){
            this.head = node;
            this.tail = node;
        }
        else if (index == 0){
            this.prepand(value);
        }
        else if (index >= this.size){
            this.add(value);
        }
        else{
            Node<E> leadingNode = traverseToIndex(index - 1);
            node.next = leadingNode.next;
            leadingNode.next = node;
            this.size++;
        }

    }

    public CustomLinkedList<E> reverse(){
        // 1 -> 2 -> 3
        // 1 -> prepand 2 -> prepand 3
        CustomLinkedList<E> reversedLinkedList = new CustomLinkedList<E>();
        Node<E> curNode = this.head;
        while(curNode != null){
            reversedLinkedList.prepand(curNode.data);
            curNode = curNode.next;
        }
        return reversedLinkedList;
    }

    public CustomLinkedList<E> reverse2(){
        // 1 -> 2 -> 3
        // 1 -> prepand 2 -> prepand 3

        // no need to reverse
        if (this.head == null || this.head.next == null){
            return this;
        }
        this.tail = this.head;
        Node<E> first = this.head;
        Node<E> second = first.next;
        // 1 -> 2 -> 3
        // 1 <- 2 -- 3
        while(second != null){
            Node<E> tmp = second.next;
            second.next = first;
            first = second;
            second = tmp;
        }
        this.head.next = null;
        this.head = first;
        return this;
    }

    public void remove(E value){
        Node<E> currNode = this.head;
        Node<E> prevNode = null;
        while(currNode != null){
            if (currNode.data == value){
                // no new node after node to be deleted
                if (currNode.next == null){
                    // currNode is head
                    if (prevNode == null){
                        this.head = null;
                        this.tail = null;
                    }
                    else{
                        prevNode.next = null;
                        this.tail = prevNode;
                    }
                }
                else{
                    if (prevNode == null){
                        this.head = currNode.next;
                    }
                    else{
                        prevNode.next = currNode.next;
                    }

                }
                size--;
                return;
            }
            prevNode = currNode;
            currNode = currNode.next;
        }
    }

    public static void main(String[] args){
        CustomLinkedList<Integer> list = new CustomLinkedList<>();
        list.add(1);
        list.add(2);
        list.insert(30, 3);
        list.prepand(0);
        System.out.println(list.reverse2());
        System.out.println(list);
    }
}
