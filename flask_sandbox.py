from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Menu items
menu_items = {
    "Burgers": [("Classic Burger", 120), ("Cheese Burger", 150), ("Double Patty Burger", 200)],
    "Drinks": [("Coke", 50), ("Pepsi", 50), ("Lemonade", 70)],
    "Sides": [("Fries", 80), ("Nuggets", 100), ("Onion Rings", 90)]
}

cart = []
hovered_index = 0  # Track hovered item index

@app.route("/")
def home():
    return render_template("menu.html", menu=menu_items, cart=cart)

@app.route("/add-to-cart", methods=["POST"])
def add_to_cart():
    data = request.json
    item_name = data.get("item")
    item_price = data.get("price")
    cart.append((item_name, item_price))
    return jsonify({"message": f"Added {item_name} to cart!", "cart": cart})

@app.route("/gesture", methods=["POST"])
def handle_gesture():
    global hovered_index
    data = request.json
    gesture = data.get("gesture")

    menu_list = sum(menu_items.values(), [])  # Flatten menu
    menu_length = len(menu_list)

    if gesture == "Swipe Down":
        hovered_index = (hovered_index + 1) % menu_length
    elif gesture == "Swipe Up":
        hovered_index = (hovered_index - 1) % menu_length
    elif gesture == "Select Item":
        cart.append(menu_list[hovered_index])

    return jsonify({"hovered": hovered_index})

@app.route("/get-hover", methods=["GET"])
def get_hover():
    return jsonify({"hovered": hovered_index})

@app.route("/checkout")
def checkout():
    total = sum(price for _, price in cart)
    cart.clear()
    return jsonify({"message": f"Checkout successful! Total: ₹{total}"})

if __name__ == "__main__":
    app.run(debug=True)