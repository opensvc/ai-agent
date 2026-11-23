package agent

// changesInstruction is added to the system prompt of every turn: what a call
// that changes state asks of the model, whoever asked it and from where, the
// page of the assistant or the dock over a list. A model asked to copy long
// lists of ids forgets some and makes others up, and counts them wrong: the
// records go by what a tool resolves, and the totals by what the results
// confirm. A model also asked in its answer to confirm a change the agent
// then had the user confirm again: the agent asks for a destructive call
// itself.
//
// A model also called a change without a word, the reason of the change and
// the answer to a question of the same message coming after the
// confirmation, once the call ran: on dev5 on 2026-11-23, GPT-6 Luna wrote no
// text beside 11 changes of 11 it had confirmed, Claude Haiku none beside 157
// of 206. The text of a response streams before its calls, so before their
// confirmation. Told to write it, both models explained the change before
// calling it, 33 of 33 over om3-answer-then-act, om3-explain-then-act and the
// action scenarios of the om3 triage, 3 runs each, and 12 of 12 again with
// this text. Told to answer a question of the message first, in full, and
// not to repeat it after the call, on om3-answer-then-act, 3 runs each:
// GPT-6 Luna answered first 3 times of 3, but repeated its answer after the
// call 3 times of 3; Claude Haiku answered first once, after the confirmation
// otherwise. The prompt alone does not order these.
const changesInstruction = "A change asked of several records, such as the rows the user ticked or the records a criterion selects, changes those records, all of them and no other: " +
	"one call of a tool that takes several, or one call for each, in one response when you can. " +
	"When the user's message also asks a question, answer it first, in full. " +
	"Then, in the same response, before the calls that change state, say in one or two sentences what leads to them and what they will change: for several records, their count and what selects them, never their list, a set a criterion gives, such as a filter or a filterset, counted before the change. " +
	"The user reads it before the calls run, beside their confirmation when they take one. " +
	"That text is not a question: never ask whether to make a change the user asked, call its tool. A tool that carries a risk asks the user to confirm it through the interface, showing the call, before it runs; a question of yours would ask twice. " +
	"After the calls, do not repeat what you wrote before them: report their result, and what is left of what the user asked. " +
	"When the records asked cannot be told exactly, ask rather than act on a wider set, and never make up an id or a name. " +
	"Report what the tool results confirm, their counts added up, never more: a call that failed or was rejected changed nothing."
