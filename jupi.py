import json
import os
import re
import urllib.request


sample_notes = """Website revamp kickoff
Alex: What are the main features you want on the new landing page?
Client: We need a modern hero section, customer testimonials, and a simple design.
Alex: Would you prefer Calendly integration or a custom-built solution?
Client: Calendly works fine and keeps it simple.
Alex: Perfect, I'll prepare the initial wireframes by Wednesday."""


def offline_extract(notes):
    """Extract lines in the format Name: task by deadline."""

    actions = []

    for line in notes.splitlines():

        # Identify the speaker
        match = re.match(r"^\s*([^:]+):\s*(.+)$", line)

        if not match:
            continue

        name = match.group(1).strip()
        sentence = match.group(2).strip()

        # Find action statements with deadlines
        action_pattern = (
            r"(?:I'll|I will|will)\s+"
            r"(.+?)\s+by\s+"
            r"([A-Za-z]+(?:\s+\d{1,2})?)"
            r"[.!]?$"
        )

        action_match = re.search(
            action_pattern,
            sentence,
            re.IGNORECASE
        )

        if action_match:
            task = action_match.group(1).strip()
            deadline = action_match.group(2).strip()

            actions.append({
                "name": name,
                "task": task,
                "deadline": deadline
            })

    return actions


def save_actions(actions):
    """Save extracted actions into a JSON file."""

    with open("actions.json", "w") as file:
        json.dump(actions, file, indent=4)

    print("\nActions saved successfully to actions.json")


def display_actions(actions):
    """Display extracted action items."""

    if not actions:
        print("No action items found.")
        return

    print("\nEXTRACTED ACTION ITEMS")
    print("-" * 40)

    for action in actions:
        print(f"Name: {action['name']}")
        print(f"Task: {action['task']}")
        print(f"Deadline: {action['deadline']}")
        print("-" * 40)


def main():
    print("MEETING NOTES ACTION EXTRACTOR")
    print("=" * 40)

    actions = offline_extract(sample_notes)

    display_actions(actions)

    save_actions(actions)


if __name__ == "__main__":
    main()
