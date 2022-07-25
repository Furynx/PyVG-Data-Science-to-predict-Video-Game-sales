"""
Config file for Streamlit App
"""

from member import Member

TITLE = "PyVGS App"

TEAM_MEMBERS = [
    Member(
	name= "Alexis Terrasse",
	linkedin_url= "https://www.linkedin.com/in/alexis-terrasse-2a85b81bb/",
	github_url= "https://github.com/Furynx",
	),
    Member("Henri-François Mole"),
    Member("Hsan Drissi",
    linkedin_url= "https://www.linkedin.com/in/drissih/",
	github_url= "https://github.com/hsndrissi",
    ),
    Member(
        name="Stephane Lelievre",
        linkedin_url="https://www.linkedin.com/in/st%C3%A9phane-lelievre-41a22b12/",
        github_url="https://github.com/sleli31",
    ),
]

PROMOTION = "Promotion formation continue Data Scientist - Octobre 2021"
