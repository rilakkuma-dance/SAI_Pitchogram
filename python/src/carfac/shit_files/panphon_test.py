import panphon
import panphon.distance

ft = panphon.FeatureTable()

dist = panphon.distance.Distance()
max_dist = 24 * max(4, 6)
example = dist.weighted_feature_edit_distance_div_maxlen(["ʃieʃie"], ["ɕiɛɕiɛ"])
example2 = dist.weighted_feature_edit_distance_div_maxlen(["ʃeʃe"], ["bombad"])
similarity = (1 - (example / 6))
similarity2 = (1 - (example2 / 6)) * 100

example3 = dist.weighted_feature_edit_distance_div_maxlen(["e"], ["ɛ"])
example4 = dist.weighted_feature_edit_distance_div_maxlen(["ʃ"], ["ɕ"])

print(f"Example distance: {example}, similarity: {similarity:.2f}%")
print(f"Example distance: {example2}, similarity: {similarity2:.2f}%")
print(f"Example distance: {example3}, {example4}")