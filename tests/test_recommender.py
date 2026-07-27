from floter_ai.recommender import RecommendationService


def test_recommendation_service_returns_top_k_results():
    service = RecommendationService()

    recommendations = service.get_recommendations("00029", top_k=5)

    assert len(recommendations) == 5
    assert all(item.user_id != "00029" for item in recommendations)
    assert all(0 <= item.score <= 1 for item in recommendations)


def test_recommendation_scores_are_sorted_descending():
    service = RecommendationService()

    recommendations = service.get_recommendations("00029", top_k=10)
    scores = [item.score for item in recommendations]

    assert scores == sorted(scores, reverse=True)


def test_recommendation_service_rejects_unknown_user():
    service = RecommendationService()

    try:
        service.get_recommendations("not-a-user", top_k=5)
    except KeyError as exc:
        assert "Unknown user_id" in str(exc)
    else:
        raise AssertionError("Expected KeyError for unknown user_id")


def test_recommendation_service_returns_results_for_sample_without_saving_user():
    service = RecommendationService()
    row_count_before = len(service.users_df)

    recommendations, ignored_interests = service.get_recommendations_for_sample(
        user_id="new_user_001",
        age=28,
        city="Mumbai",
        country="India",
        interests=["Technology", "Travel", "Technology", "Unknown Interest"],
        top_k=5,
    )

    assert len(recommendations) == 5
    assert ignored_interests == ["Unknown Interest"]
    assert len(service.users_df) == row_count_before
    assert all(item.user_id != "new_user_001" for item in recommendations)
    assert all(0 <= item.score <= 1 for item in recommendations)


def test_recommendation_service_rejects_sample_with_no_known_interests():
    service = RecommendationService()

    try:
        service.get_recommendations_for_sample(
            user_id="new_user_001",
            age=28,
            city="Mumbai",
            country="India",
            interests=["Unknown Interest"],
            top_k=5,
        )
    except ValueError as exc:
        assert "known interest" in str(exc)
    else:
        raise AssertionError("Expected ValueError for sample with no known interests")
