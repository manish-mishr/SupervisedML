# Use this class to represent a single element of the LinkedList
class Node:
    next_node = None
    prev_node = None
    data = None

    def __init__(self, data):
        self.data = data


# Use this class to implement the LinkedList
# A Linked List is a data structure made of Nodes, which each contain a reference to the next Node in the list
# You may choose to construct a Doubly Linked List, where each Node also contains a reference to the
# previous Node in the list, but doing so is not required.
# The important idea is that Nodes don't know about all of the items in the list, only the next
# (and in a Doubly Linked List, the previous) element.
# You should not have a table of all of the elements in this structure.
class LinkedList:
    sentinel = Node(None)

    def __init__(self):
        self.sentinel.next_node = self.sentinel
        self.sentinel.prev_node = self.sentinel

    # Use this function to insert a new element in the LinkedList
    # Like a stack, elements should be added to the front (called the sentinel) of the LinkedList.
    # None should not be allowed to be added to the list.
    def insert(self, data):
        if data is None:
            return
        n = Node(data)
        n.prev_node = self.sentinel
        n.next_node = self.sentinel.next_node
        self.sentinel.next_node.prev_node = n
        self.sentinel.next_node = n

    # Use this function to find a Node with the specified data in the LinkedList
    # If the end of the list is reached, and the data are not found, return None.
    # If the list contains multiple Nodes with the requested Data, you only need to return the first one you find.
    def find(self, data):
        n = self.sentinel.next_node
        while n.data is not None:
            if n.data == data:
                return n
            n = n.next_node
        return None

    # Use this function to remove the specified Node from the LinkedList
    # Note that this function takes a Node (not data) as an argument.
    # Once the Node is removed, the list should still be able to be traversed - references will
    # need to be updated to accomplish this task!
    # Think carefully about what needs to happen if the Node at the front of the list is removed,
    # or if this node is the last Node in the list.
    # noinspection PyMethodMayBeStatic
    def remove(self, node):
        node.prev_node.next_node = node.next_node
        node.next_node.prev_node = node.prev_node


# Use this class to implement the Stack
# A stack is a data structure from which items can be added and removed.
# Just like a stack of books, plates, lunch trays, or anything else, items are both removed from,
# and added to, the "top" of the stack.
class Stack:
    ll = LinkedList()

    # Use this function to insert a new element in the Stack
    # None should never be able to be pushed into a stack.
    def push(self, item):
        self.ll.insert(item)

    # Use this function to remove and return the top element from the Stack
    # The item popped from a stack is always the last item added (pushed) to the stack.
    # If no items are present in the stack, pop() returns None.
    def pop(self):
        n = self.ll.sentinel.next_node
        self.ll.remove(n)
        return n.data

    # Use this function to read the top element in the Stack without removing it.
    # This function returns the same value as pop(), but does not change the stack.
    # Repeated, consecutive calls to peek() will always return the same value.
    def peek(self):
        return self.ll.sentinel.next_node.data


# Use this class to implement the FIFO (First In First Out) Queue
# A FIFO Queue is a data structure where the first item added to the data structure is the first item removed.
# A queue acts like a line at the movies - the first person to line up is the first person into the theatre.
class Queue:
    ll = LinkedList()

    # Use this function to insert a new element in the Queue
    # None should never be able to be inserted into the Queue.
    def enqueue(self, item):
        self.ll.insert(item)

    # Use this function to remove and return the next element from the Queue
    # If the Queue is empty, this function should return None
    def dequeue(self):
        n = self.ll.sentinel.prev_node
        self.ll.remove(n)
        return n.data

    # Use this function to read the top element in the Queue without removing it
    # This function returns the same value as dequeue(), but does not change the stack.
    # Repeated, consecutive calls to peek() will always return the same value.
    def peek(self):
        return self.ll.sentinel.prev_node.data


mystack = Stack()
mystack.push(4)
mystack.push(5)
mystack.push(None)
mystack.push(None)
nextD = mystack.ll.sentinel.next_node
nextD2 = nextD.next_node
nextD3 = nextD2.next_node
nextD4 = nextD3.next_node
print mystack.ll.sentinel.data
print nextD2.data
print nextD3.data
print nextD4.data
# print mystack.pop()
# print mystack.pop()
# print mystack.pop()
# print mystack.pop()
# print mystack.pop()
# print mystack.pop()
# print mystack.pop()
# print mystack.peek()
