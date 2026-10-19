| feature_set | added | cv_accuracy | cv_f1 | share_of_gap_recovered |
|---|---|---|---|---|
| lite (21) |  | 0.9504 | 0.9434 | 0.0000 |
| lite + 1 | domain_registration_length | 0.9502 | 0.9433 | -0.0058 |
| lite + 1 | abnormal_url | 0.9506 | 0.9436 | 0.0091 |
| lite + 1 | popupwindow | 0.9496 | 0.9426 | -0.0380 |
| lite + 1 | age_of_domain | 0.9509 | 0.9443 | 0.0390 |
| lite + 1 | web_traffic | 0.9535 | 0.9475 | 0.1817 |
| lite + 1 | page_rank | 0.9519 | 0.9452 | 0.0798 |
| lite + 1 | google_index | 0.9492 | 0.9422 | -0.0547 |
| lite + 1 | links_pointing_to_page | 0.9542 | 0.9480 | 0.2052 |
| lite + 1 | statistical_report | 0.9501 | 0.9433 | -0.0058 |
| lite + top 2 | web_traffic, links_pointing_to_page | 0.9591 | 0.9536 | 0.4512 |
| lite + top 3 | web_traffic, links_pointing_to_page, age_of_domain | 0.9625 | 0.9576 | 0.6290 |
| lite + top 4 | web_traffic, links_pointing_to_page, age_of_domain, google_index | 0.9648 | 0.9602 | 0.7458 |
| lite + top 5 | web_traffic, links_pointing_to_page, age_of_domain, google_index, domain_registration_length | 0.9656 | 0.9611 | 0.7845 |
| lite + top 6 | web_traffic, links_pointing_to_page, age_of_domain, google_index, domain_registration_length, page_rank | 0.9689 | 0.9648 | 0.9485 |
| lite + top 7 | web_traffic, links_pointing_to_page, age_of_domain, google_index, domain_registration_length, page_rank, statistical_report | 0.9698 | 0.9658 | 0.9934 |
| lite + top 8 | web_traffic, links_pointing_to_page, age_of_domain, google_index, domain_registration_length, page_rank, statistical_report, popupwindow | 0.9689 | 0.9648 | 0.9485 |
| full (30) | all 9 | 0.9699 | 0.9659 | 1.0000 |
