d = {1: 4, 3: 5, 12: 8}
items = [(1, 4), (3, 5), (12, 8)]


best_t= -1
best_user_id = None
for user_id, t in items:
    if t > best_t:
        best_t = t
        best_user_id = user_id






