
import React from "react";

function ExpressionResult({ result, loading, error }) {
  if (loading) {
    return (
      <div className="expression-result">
        Analyzing facial expression...
      </div>
    );
  }

  if (error) {
    return (
      <div className="expression-result">
        Unable to detect expression: {error}
      </div>
    );
  }

  if (!result) return null;

  return (
    <div className="expression-result">
      <h3>Facial Expression Result</h3>

      <p>
        Detected expression: <strong>{result.emotion}</strong>
      </p>

      <p>
        Confidence: <strong>{result.confidence}%</strong>
      </p>

      <h4>Expression probabilities</h4>

      {Object.entries(result.probabilities).map(
        ([emotion, probability]) => (
          <p key={emotion}>
            {emotion}: {probability}%
          </p>
        )
      )}

      <small>
        This is an approximate visual prediction, not a
        measurement of the person's actual emotional state.
      </small>
    </div>
  );
}

export default ExpressionResult;
