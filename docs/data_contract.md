# SmartDine Data Contract (v1)

**Purpose**  
This contract defines the schemas, locations, and validation rules for SmartDine’s raw inputs and processed outputs. It is the single source of truth for Week 1–2 data handling.

## 1) Entities & Scope
- **Interaction row** (atomic unit): a user’s review/rating event for a restaurant.
- **Restaurant** metadata may be derived later; not required for Week 1.
- **Images** are referenced via `pics` at the interaction level and expanded in an `image_manifest`.

## 2) Required Columns (Interaction Schema)
| Field            | Type            | Required | Description                                                                                 | Constraints                                  |
|------------------|-----------------|----------|---------------------------------------------------------------------------------------------|----------------------------------------------|
| `business_id`    | string          | Yes      | Unique restaurant identifier in dataset                                                     | Non-empty; canonicalized string              |
| `user_id`        | string          | Yes      | Unique user identifier in dataset                                                           | Non-empty; canonicalized string              |
| `rating`         | integer (1–5)   | Yes      | User-provided star rating                                                                   | ∈ {1,2,3,4,5}                                |
| `review_text`    | string          | Yes      | The textual review content                                                                  | May be empty **only if** `rating` exists     |
| `pics`           | list[string]    | Yes      | List of image IDs associated with this interaction                                          | List (can be empty); elements are non-empty  |
| `history_reviews`| list[object/str]| Yes      | Prior review identifiers/objects for this user (format as provided by source)               | List (can be empty)                          |

**Notes**
- `review_text` empty is permitted (some ratings have no text), but presence of the field itself is required.
- `pics` is a **list of image IDs**, not URLs. URLs/paths live in the `image_manifest`.

## 3) Processed Outputs (Parquet) — Locations & Schemas

All processed files live **locally** (not tracked by Git) at:
