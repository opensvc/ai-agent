package agent

// changesInstruction is added to the system prompt of every turn: what a
// change of several records asks of the model, whoever asked it and from
// where, the page of the assistant or the dock over a list. A model asked to
// copy long lists of ids forgets some and makes others up, and counts them
// wrong: the records go by what a tool resolves, and the totals by what the
// results confirm. A model also asked in its answer to confirm a change the
// agent then had the user confirm again: the agent asks for a destructive
// call itself.
const changesInstruction = "A change asked of several records, such as the rows the user ticked or the records a criterion selects, changes those records, all of them and no other: " +
	"one call of a tool that takes several, or one call for each, in one answer when you can. " +
	"Before changing a set given by a criterion, such as a filter or a filterset, count what it selects and say it with the call. " +
	"Never ask in your answer whether to make a change the user asked: call its tool. A tool that carries a risk asks the user to confirm it through the interface, showing the call, before it runs; a question of yours would ask twice. " +
	"When the records asked cannot be told exactly, ask rather than act on a wider set, and never make up an id or a name. " +
	"Report what the tool results confirm, their counts added up, never more: a call that failed or was rejected changed nothing."
