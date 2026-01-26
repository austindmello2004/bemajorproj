# Streamlit Web Application
import asyncio
import json
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

from agents.chat_agent import PolicyChatAgent
from agents.compliance_validator_agent import ComplianceValidatorAgent
from agents.orchestrator import OrchestratorAgent
from agents.policy_agent import PolicyRuleAgent
from agents.schedule_generator_agent import ScheduleGeneratorAgent
from db.compliance_database import ComplianceDatabase
from tools.rag import build_chunks_with_embeddings
from utils.exceptions import ResumeProcessingError
from utils.logger import setup_logger
from utils.parsers import load_policy_file, parse_schedule_file

# Configure Streamlit page
st.set_page_config(
    page_title="AI Optimization Agency",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize logger
logger = setup_logger()


@st.cache_resource
def get_compliance_db() -> ComplianceDatabase:
    return ComplianceDatabase()


@st.cache_resource
def get_policy_agent() -> PolicyRuleAgent:
    return PolicyRuleAgent()


@st.cache_resource
def get_validator_agent() -> ComplianceValidatorAgent:
    return ComplianceValidatorAgent()


@st.cache_resource
def get_schedule_generator_agent() -> ScheduleGeneratorAgent:
    return ScheduleGeneratorAgent()


@st.cache_resource
def get_chat_agent() -> PolicyChatAgent:
    return PolicyChatAgent()

# Custom CSS
st.markdown(
    """
    <style>
        .stProgress .st-bo {
            background-color: #00a0dc;
        }
        .success-text {
            color: #00c853;
        }
        .warning-text {
            color: #ffd700;
        }
        .error-text {
            color: #ff5252;
        }
        .st-emotion-cache-1v0mbdj.e115fcil1 {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
        }
    </style>
""",
    unsafe_allow_html=True,
)


async def process_resume(file_path: str) -> dict:
    """Process resume through the AI recruitment pipeline"""
    try:
        orchestrator = OrchestratorAgent()
        resume_data = {
            "file_path": file_path,
            "submission_timestamp": datetime.now().isoformat(),
        }
        return await orchestrator.process_application(resume_data)
    except Exception as e:
        logger.error(f"Error processing resume: {str(e)}")
        raise


def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file and return the file path"""
    try:
        # Create uploads directory if it doesn't exist
        save_dir = Path("uploads")
        save_dir.mkdir(exist_ok=True)

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = save_dir / f"resume_{timestamp}_{uploaded_file.name}"

        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return str(file_path)
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        raise


def render_resume_analysis():
    st.header("📄 Resume Analysis")
    st.write("Upload a resume to get AI-powered insights and job matches.")

    uploaded_file = st.file_uploader(
        "Choose a PDF resume file",
        type=["pdf"],
        help="Upload a PDF resume to analyze",
    )

    if uploaded_file:
        try:
            with st.spinner("Saving uploaded file..."):
                file_path = save_uploaded_file(uploaded_file)

            st.info("Resume uploaded successfully! Processing...")

            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("Analyzing resume...")
                progress_bar.progress(25)

                result = asyncio.run(process_resume(file_path))

                if result.get("status") == "completed":
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")

                    tab1, tab2, tab3, tab4 = st.tabs(
                        ["📊 Analysis", "💼 Job Matches", "🎯 Screening", "💡 Recommendation"]
                    )

                    with tab1:
                        st.subheader("Skills Analysis")
                        st.write(result.get("analysis_results", {}).get("skills_analysis", {}))
                        st.metric(
                            "Confidence Score",
                            f"{result.get('analysis_results', {}).get('confidence_score', 0):.0%}",
                        )

                    with tab2:
                        st.subheader("Matched Positions")
                        matches = result.get("job_matches", {}).get("matched_jobs", [])
                        if not matches:
                            st.warning("No suitable positions found.")
                        seen_titles = set()
                        for job in matches:
                            if job.get("title") in seen_titles:
                                continue
                            seen_titles.add(job.get("title"))
                            with st.container():
                                col1, col2, col3 = st.columns([2, 1, 1])
                                with col1:
                                    st.write(f"**{job.get('title','N/A')}**")
                                with col2:
                                    st.write(f"Match: {job.get('match_score', 'N/A')}")
                                with col3:
                                    st.write(f"📍 {job.get('location', 'N/A')}")
                            st.divider()

                    with tab3:
                        st.subheader("Screening Results")
                        st.metric(
                            "Screening Score",
                            f"{result.get('screening_results', {}).get('screening_score', 0)}%",
                        )
                        st.write(result.get("screening_results", {}).get("screening_report", ""))

                    with tab4:
                        st.subheader("Final Recommendation")
                        st.info(
                            result.get("final_recommendation", {}).get("final_recommendation", ""),
                            icon="💡",
                        )

                    output_dir = Path("results")
                    output_dir.mkdir(exist_ok=True)
                    output_file = output_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    with open(output_file, "w") as f:
                        f.write(str(result))
                    st.success(f"Results saved to: {output_file}")
                else:
                    st.error(
                        f"Process failed at stage: {result.get('current_stage')}\n"
                        f"Error: {result.get('error', 'Unknown error')}"
                    )
            except Exception as e:  # noqa: E722
                st.error(f"Error processing resume: {str(e)}")
                logger.error(f"Processing error: {str(e)}", exc_info=True)
            finally:
                try:
                    os.remove(file_path)
                except Exception as e:  # noqa: E722
                    logger.error(f"Error removing temporary file: {str(e)}")
        except Exception as e:  # noqa: E722
            st.error(f"Error handling file upload: {str(e)}")
            logger.error(f"Upload error: {str(e)}", exc_info=True)


def render_policy_rules(policy_agent: PolicyRuleAgent, compliance_db: ComplianceDatabase):
    st.header("📑 Policy Understanding & Rule Extraction")
    st.write("Upload policy documents (PDF/DOCX) to extract structured compliance rules and build a RAG index.")

    uploaded_policy = st.file_uploader(
        "Upload a policy document",
        type=["pdf", "docx", "txt"],
        help="Baseline policy.pdf is supported; upload additional policies as needed.",
    )

    if uploaded_policy:
        try:
            with st.spinner("Reading policy document..."):
                policy_text = load_policy_file(uploaded_policy)
            st.success("Policy text extracted. Generating rules and embeddings...")

            policy_id = compliance_db.add_policy(uploaded_policy.name, uploaded_policy.name, policy_text)

            chunks = build_chunks_with_embeddings(policy_agent, policy_text)
            compliance_db.add_policy_chunks(policy_id, chunks)

            rule_response = asyncio.run(
                policy_agent.run([
                    {"role": "user", "content": policy_text}
                ])
            )
            rules = rule_response.get("rules", [])
            compliance_db.add_policy_rules(policy_id, rules)

            st.success("Rules extracted and stored.")
            st.subheader("Extracted Rules")
            st.dataframe(pd.DataFrame(rules))

        except Exception as exc:  # noqa: E722
            st.error(f"Policy processing failed: {exc}")
            logger.error(f"Policy processing failed: {exc}", exc_info=True)

    existing = compliance_db.list_policies()
    if existing:
        st.write("Latest policies in the system:")
        st.dataframe(pd.DataFrame(existing))


def render_schedule_validation(
    compliance_db: ComplianceDatabase,
    validator_agent: ComplianceValidatorAgent,
    schedule_generator_agent: ScheduleGeneratorAgent,
):
    st.header("🗓️ Schedule Compliance & Correction")
    st.write(
        "Upload employee schedules (CSV/JSON/XLSX). The system will validate against extracted policies, "
        "highlight violations, and propose compliant schedules."
    )

    if not compliance_db.get_latest_policy_id():
        st.warning("Please upload and extract at least one policy before validating schedules.")

    schedule_file = st.file_uploader(
        "Upload schedule file",
        type=["csv", "json", "xlsx"],
        help="Use accepted/rejected schedule examples or new schedules for validation.",
    )

    if schedule_file:
        try:
            schedule_meta, schedule_items = parse_schedule_file(schedule_file)
            schedule_id = compliance_db.add_schedule(
                schedule_meta.get("name", "schedule"),
                "upload",
                schedule_meta.get("payload", {}),
                {"source_type": schedule_meta.get("source_type")},
            )
            compliance_db.add_schedule_items(schedule_id, schedule_items)

            latest_policy_id = compliance_db.get_latest_policy_id()
            policy_rules = compliance_db.list_policy_rules(latest_policy_id)

            context = {
                "schedule_items": schedule_items,
                "policy_rules": policy_rules,
            }
            validation = asyncio.run(
                validator_agent.run([
                    {"role": "user", "content": json.dumps(context)}
                ])
            )

            st.subheader("Compliance Results")
            st.metric("Compliance Score", f"{validation.get('compliance_score', 0)}")
            violations = validation.get("violations", [])
            st.write(validation.get("summary", ""))

            for v in violations:
                compliance_db.add_violation(
                    schedule_id,
                    v.get("employee_id", ""),
                    None,
                    v.get("description", ""),
                    v.get("severity", "medium"),
                    v,
                )

            if violations:
                st.warning("Violations detected")
                st.dataframe(pd.DataFrame(violations))
                generator_context = {
                    "schedule_items": schedule_items,
                    "policy_rules": policy_rules,
                    "violations": violations,
                }
                generated = asyncio.run(
                    schedule_generator_agent.run([
                        {"role": "user", "content": json.dumps(generator_context)}
                    ])
                )
                corrected_schedule = generated.get("corrected_schedule", [])
                if corrected_schedule:
                    compliance_db.add_corrected_schedule(
                        schedule_id, generated, generated.get("explanation", "")
                    )
                    st.success("Compliant schedule generated")
                    st.dataframe(pd.DataFrame(corrected_schedule))
                    _render_downloads(corrected_schedule, "corrected_schedule")
                else:
                    st.info("No corrected schedule was generated.")
            else:
                st.success("No violations detected. Schedule is compliant.")

        except Exception as exc:  # noqa: E722
            st.error(f"Schedule validation failed: {exc}")
            logger.error(f"Schedule validation failed: {exc}", exc_info=True)


def render_policy_chat(chat_agent: PolicyChatAgent, compliance_db: ComplianceDatabase):
    st.header("💬 Policy Chatbot (RAG)")
    st.write("Ask questions about policies, violations, or schedule adjustments. Uses local RAG over uploaded policies.")

    question = st.text_input("Ask a policy question")
    if question:
        latest_policy_id = compliance_db.get_latest_policy_id()
        rules = compliance_db.list_policy_rules(latest_policy_id)
        query_embedding = chat_agent.embed_text(question)
        similar_chunks = compliance_db.search_similar_chunks(query_embedding, top_k=5)
        context = [c.get("content") for c in similar_chunks]
        response = asyncio.run(
            chat_agent.run([
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": question,
                            "context": context,
                            "rules": rules,
                        }
                    ),
                }
            ])
        )
        st.write(response.get("answer", ""))
        if similar_chunks:
            st.caption("Context used:")
            for chunk in similar_chunks:
                score = chunk.get("score", 0.0)
                st.markdown(f"- {chunk.get('content')[:180]}… (score {score:.2f})")


def render_about():
    st.header("About AI Optimization Agency")
    st.write(
        """
        Welcome to AI Optimization Agency, a recruitment and workforce compliance system powered by:

        - **Ollama (llama3.2)**: Local LLM via OpenAI-compatible API
        - **Swarm-inspired multi-agents**: Specialized agents for extraction, analysis, matching, screening, compliance, and generation
        - **Streamlit**: Modern UI for resume analysis, policy ingestion, schedule validation, and chat
        """
    )


def _render_downloads(records, filename_prefix: str):
    if not records:
        return
    df = pd.DataFrame(records)
    json_bytes = json.dumps(records, indent=2).encode("utf-8")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    excel_bytes = excel_buffer.getvalue()

    st.download_button(
        label="Download JSON",
        data=json_bytes,
        file_name=f"{filename_prefix}.json",
        mime="application/json",
    )
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=f"{filename_prefix}.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download XLSX",
        data=excel_bytes,
        file_name=f"{filename_prefix}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def main():
    compliance_db = get_compliance_db()
    policy_agent = get_policy_agent()
    validator_agent = get_validator_agent()
    schedule_generator_agent = get_schedule_generator_agent()
    chat_agent = get_chat_agent()

    # Sidebar navigation
    with st.sidebar:
        st.image(
            "https://img.icons8.com/resume",
            width=50,
        )
        st.title("AI Optimization Agency")
        selected = option_menu(
            menu_title="Navigation",
            options=[
                "Resume Analysis",
                "Policies & Rules",
                "Schedules & Compliance",
                "Policy Chat",
                "About",
            ],
            icons=[
                "cloud-upload",
                "file-earmark-text",
                "calendar-check",
                "chat",
                "info-circle",
            ],
            menu_icon="cast",
            default_index=0,
        )

    if selected == "Resume Analysis":
        render_resume_analysis()
    elif selected == "Policies & Rules":
        render_policy_rules(policy_agent, compliance_db)
    elif selected == "Schedules & Compliance":
        render_schedule_validation(
            compliance_db, validator_agent, schedule_generator_agent
        )
    elif selected == "Policy Chat":
        render_policy_chat(chat_agent, compliance_db)
    elif selected == "About":
        render_about()


if __name__ == "__main__":
    main()


# == Command Line Interface (CLI) ==
# == to run use: python3 app.py resumes/sample_resume.pdf ==
# import asyncio
# import os
# import sys
# from datetime import datetime
# from rich.console import Console
# from rich.panel import Panel
# from rich.progress import Progress, SpinnerColumn, TextColumn
# from rich.table import Table
# from agents.orchestrator import OrchestratorAgent
# from utils.logger import setup_logger
# from utils.exceptions import ResumeProcessingError

# # Initialize Rich console for beautiful CLI output
# console = Console()
# logger = setup_logger()


# async def process_resume(file_path: str) -> None:
#     """Process a resume through the AI recruitment pipeline"""
#     try:
#         # Validate input file
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"Resume not found at {file_path}")

#         if not file_path.lower().endswith(".pdf"):
#             raise ValueError("Please provide a PDF resume file")

#         logger.info(f"Starting recruitment process for: {os.path.basename(file_path)}")

#         # Display welcome banner
#         console.print(
#             Panel.fit(
#                 "[bold blue]AI Recruitment Agency[/bold blue]\n"
#                 "[dim]Powered by Ollama (llama2) and Swarm Framework[/dim]",
#                 border_style="blue",
#             )
#         )

#         # Initialize orchestrator
#         orchestrator = OrchestratorAgent()

#         # Prepare resume data
#         resume_data = {
#             "file_path": file_path,
#             "submission_timestamp": datetime.now().isoformat(),
#         }

#         # Process with progress indication
#         with Progress(
#             SpinnerColumn(),
#             TextColumn("[progress.description]{task.description}"),
#             console=console,
#         ) as progress:
#             task = progress.add_task("[cyan]Processing resume...", total=None)
#             result = await orchestrator.process_application(resume_data)
#             progress.update(task, completed=True)

#         if result["status"] == "completed":
#             logger.info("Resume processing completed successfully")

#             # Create results table
#             table = Table(
#                 title="Analysis Summary", show_header=True, header_style="bold magenta"
#             )
#             table.add_column("Category", style="cyan")
#             table.add_column("Details", style="white")

#             # Add analysis results
#             table.add_row(
#                 "Skills Analysis", str(result["analysis_results"]["skills_analysis"])
#             )
#             table.add_row(
#                 "Confidence Score",
#                 f"{result['analysis_results']['confidence_score']:.2%}",
#             )

#             console.print("\n", table)

#             # Display job matches
#             matches_table = Table(
#                 title="Job Matches", show_header=True, header_style="bold green"
#             )
#             matches_table.add_column("Position", style="cyan")
#             matches_table.add_column("Match Score", style="white")
#             matches_table.add_column("Location", style="white")

#             for job in result["job_matches"]["matched_jobs"]:
#                 matches_table.add_row(
#                     job["title"],
#                     f"{job.get('match_score', 'N/A')}",
#                     job.get("location", "N/A"),
#                 )

#             console.print("\n", matches_table)

#             # Display screening results
#             console.print(
#                 Panel(
#                     f"[bold]Screening Score:[/bold] {result['screening_results']['screening_score']}%\n\n"
#                     f"{result['screening_results']['screening_report']}",
#                     title="Screening Results",
#                     border_style="green",
#                 )
#             )

#             # Display final recommendation
#             console.print(
#                 Panel(
#                     result["final_recommendation"]["final_recommendation"],
#                     title="Final Recommendation",
#                     border_style="blue",
#                 )
#             )

#             # Save results to file
#             output_dir = "results"
#             if not os.path.exists(output_dir):
#                 os.makedirs(output_dir)

#             output_file = os.path.join(
#                 output_dir, f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
#             )

#             with open(output_file, "w") as f:
#                 f.write("AI Recruitment Analysis Results\n")
#                 f.write("=" * 50 + "\n\n")
#                 f.write(f"Resume: {os.path.basename(file_path)}\n")
#                 f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
#                 f.write("Analysis Summary\n")
#                 f.write("-" * 20 + "\n")
#                 f.write(
#                     f"Skills Analysis: {result['analysis_results']['skills_analysis']}\n"
#                 )
#                 f.write(
#                     f"Confidence Score: {result['analysis_results']['confidence_score']:.2%}\n\n"
#                 )
#                 f.write("Job Matches\n")
#                 f.write("-" * 20 + "\n")
#                 for job in result["job_matches"]["matched_jobs"]:
#                     f.write(f"\nPosition: {job['title']}\n")
#                     f.write(f"Match Score: {job.get('match_score', 'N/A')}\n")
#                     f.write(f"Location: {job.get('location', 'N/A')}\n")
#                 f.write("\nScreening Results\n")
#                 f.write("-" * 20 + "\n")
#                 f.write(f"Score: {result['screening_results']['screening_score']}%\n")
#                 f.write(
#                     f"Report: {result['screening_results']['screening_report']}\n\n"
#                 )
#                 f.write("Final Recommendation\n")
#                 f.write("-" * 20 + "\n")
#                 f.write(str(result["final_recommendation"]["final_recommendation"]))

#             console.print(f"\n[green]✓[/green] Results saved to: {output_file}")

#         else:
#             error_msg = f"Process failed at stage: {result['current_stage']}"
#             if "error" in result:
#                 error_msg += f"\nError: {result['error']}"
#             logger.error(error_msg)
#             console.print(f"\n[red]✗[/red] {error_msg}")

#     except FileNotFoundError as e:
#         logger.error(f"File error: {str(e)}")
#         console.print(f"[red]Error:[/red] {str(e)}")
#     except ValueError as e:
#         logger.error(f"Validation error: {str(e)}")
#         console.print(f"[red]Error:[/red] {str(e)}")
#     except ResumeProcessingError as e:
#         logger.error(f"Processing error: {str(e)}")
#         console.print(f"[red]Error:[/red] {str(e)}")
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}", exc_info=True)
#         console.print(f"[red]✗ An unexpected error occurred:[/red] {str(e)}")


# def main():
#     """Main entry point for the AI recruitment system"""
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="AI Recruitment Agency - Resume Processing System",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#     python main.py path/to/resume.pdf
#     python main.py --help
#         """,
#     )

#     parser.add_argument("resume_path", help="Path to the PDF resume file to process")

#     parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

#     args = parser.parse_args()

#     if args.verbose:
#         console.print("[yellow]Running in verbose mode[/yellow]")

#     try:
#         asyncio.run(process_resume(args.resume_path))
#     except KeyboardInterrupt:
#         console.print("\n[yellow]Process interrupted by user[/yellow]")
#         sys.exit(1)


# if __name__ == "__main__":
#     main()
