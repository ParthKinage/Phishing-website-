| method | top5 | top10_overlap_with_consensus |
|---|---|---|
| builtin_decision_tree | sslfinal_state, url_of_anchor, web_traffic, links_in_tags, having_sub_domain | 9 |
| builtin_random_forest | sslfinal_state, url_of_anchor, web_traffic, having_sub_domain, links_in_tags | 9 |
| builtin_xgboost | sslfinal_state, url_of_anchor, prefix_suffix, shortining_service, sfh | 7 |
| perm_decision_tree | sslfinal_state, url_of_anchor, links_in_tags, web_traffic, having_sub_domain | 9 |
| perm_random_forest | sslfinal_state, url_of_anchor, prefix_suffix, web_traffic, links_in_tags | 10 |
| perm_svm | url_of_anchor, sslfinal_state, prefix_suffix, web_traffic, having_sub_domain | 9 |
| perm_xgboost | url_of_anchor, sslfinal_state, web_traffic, prefix_suffix, links_in_tags | 9 |
| shap_random_forest | sslfinal_state, url_of_anchor, web_traffic, having_sub_domain, links_in_tags | 9 |
